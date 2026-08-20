from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from scipy.signal import find_peaks

from app.domain.models import RepContext, PhaseSegment
from app.domain.enums import ValidationStatus, SignalSource


# ═══════════════════════════════════════════════════════
# DetectedRep (V7.1: 增加生物力学元数据)
# ═══════════════════════════════════════════════════════

@dataclass
class DetectedRep:
    start_frame: int
    bottom_frame: int
    end_frame: int
    total_rom: float
    duration: float
    eccentric_duration: float = 0.0
    concentric_duration: float = 0.0
    closure: float = 0.0
    score: float = 0.0
    # ── V7.1 新增字段 ──
    bottom_prominence: float = 0.0
    interior_valley_count: int = 0
    concentric_reversal_count: int = 0
    eccentric_reversal_count: int = 0
    phases: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════
# ScoredCandidate (V7.1: 分项评分)
# ═══════════════════════════════════════════════════════

@dataclass
class ScoredCandidate:
    rep: DetectedRep
    score: float = 0.0
    hard_rejected: bool = False
    hard_reject_reason: Optional[str] = None
    sub_scores: Dict[str, float] = field(default_factory=dict)  # V7.1: 分项
    in_final_chain: bool = False


class CycleRepDetector:
    """
    V7.2.7: + Internal Bridge Ghost Filter
    V7.2.6: Tail Ghost + Pause Merge + Shared Boundary Gate + Isolated Rest

    Pipeline:
      1. Multi-Candidate Generation
      2. Biomechanical Scoring (closure强化, reversal deadband)
      3. DP Chain Selection (Shared Boundary Gate)
      4. Pause Split Merge
      5. Tail Ghost Filter
      6. Rerack + Isolated Rest Filter
    """

    def __init__(self, fps: float, config: Optional[Dict[str, Any]] = None):
        self.fps = fps
        cfg = config or {}

        # ── 极值检测 ──
        self.min_distance_sec = cfg.get("min_distance_sec", 0.55)       # V7.1
        self.min_prominence = cfg.get("min_prominence", 10.0)           # V7.2: 8→10

        # ── Hard reject（仅极端情况）──
        self.absolute_min_duration_s = cfg.get("absolute_min_duration_s", 0.40)   # V7.1
        self.absolute_max_duration_s = cfg.get("absolute_max_duration_s", 3.20)   # V7.1
        self.absolute_min_rom_deg = cfg.get("absolute_min_rom_deg", 18.0)         # V7.1

        # ── 评分参数 ──
        self.ideal_min_rom_deg = cfg.get("ideal_min_rom_deg", 50.0)
        self.ideal_max_rom_deg = cfg.get("ideal_max_rom_deg", 130.0)
        self.ideal_closure_deg = cfg.get("ideal_closure_deg", 20.0)
        self.max_closure_deg = cfg.get("max_closure_deg", 55.0)                   # V7.1
        self.min_cycle_duration_s = cfg.get("min_cycle_duration_s", 0.5)
        self.max_cycle_duration_s = cfg.get("max_cycle_duration_s", 3.0)          # V7.1: 5.0→3.0

        # ── Phase Duration (V7.1: 收紧) ──
        self.min_eccentric_duration_s = cfg.get("min_eccentric_duration_s", 0.35) # V7.1: 0.20→0.35
        self.min_concentric_duration_s = cfg.get("min_concentric_duration_s", 0.30) # V7.1: 0.20→0.30

        # ── Spacing (V7.1: 收紧) ──
        self.min_bottom_to_bottom_frames = cfg.get("min_bottom_to_bottom_frames", 30) # V7.1: 20→30

        # ── V7.1: Candidate Generation ──
        self.max_candidates_per_valley = cfg.get("max_candidates_per_valley", 2)  # V7.1: 3→2

        # ── V7.1: DP ──
        self.min_chain_score = cfg.get("min_chain_score", 0.35)                   # V7.1
        self.rep_selection_cost = cfg.get("rep_selection_cost", 0.25)             # V7.2: 0.20→0.25
        self.target_cycle_duration_s = cfg.get("target_cycle_duration_s", 2.0)    # V7.2: 1.2→2.0

        # ── V7.1: Prominence 评分门槛 ──
        self.prominence_excellent = cfg.get("prominence_excellent", 20.0)
        self.prominence_good = cfg.get("prominence_good", 14.0)
        self.prominence_min = cfg.get("prominence_min", 8.0)

        # ── Rerack Filter ──
        self.rerack_gap_multiplier = cfg.get("rerack_gap_multiplier", 3.0)
        self.rerack_min_gap_s = cfg.get("rerack_min_gap_s", 4.0)
        self.rerack_edge_frames = cfg.get("rerack_edge_frames", 30)
        self.rerack_rom_ratio = cfg.get("rerack_rom_ratio", 0.85)
        self.rerack_long_next_gap_s = cfg.get("rerack_long_next_gap_s", 3.0)

        # ── V7.2.4: Shared Boundary Gate 参数（硬编码）──

        # ── V7.2.5: Pause Split Merge 参数（硬编码）──

        # ── V7.2.6: Tail Ghost Filter ──
        self.tail_ghost_enabled = cfg.get("tail_ghost_enabled", True)
        self.tail_ghost_max_gap_sec = cfg.get("tail_ghost_max_gap_sec", 1.2)
        self.tail_ghost_min_strong_score = cfg.get("tail_ghost_min_strong_score", 0.75)
        self.tail_ghost_max_weak_score = cfg.get("tail_ghost_max_weak_score", 0.65)
        self.tail_ghost_rom_ratio = cfg.get("tail_ghost_rom_ratio", 0.72)
        self.tail_ghost_prom_ratio = cfg.get("tail_ghost_prom_ratio", 0.55)

        # ── V7.2.7: Internal Bridge Ghost Filter ──
        self.bridge_enabled = cfg.get("bridge_enabled", True)
        self.bridge_boundary_frames = cfg.get("bridge_boundary_frames", 2)
        self.bridge_max_score = cfg.get("bridge_max_score", 0.60)
        self.bridge_max_prominence = cfg.get("bridge_max_prominence", 15.0)
        self.bridge_min_score_gap = cfg.get("bridge_min_score_gap", 0.20)
        self.bridge_min_strong_score = cfg.get("bridge_min_strong_score", 0.75)

        # 版本审计
        print(
            f"🧬 CycleRepDetector V7.2.7 | "
            f"prom={self.min_prominence} "
            f"min_ecc={self.min_eccentric_duration_s} "
            f"min_con={self.min_concentric_duration_s} "
            f"bridge={self.bridge_enabled} "
            f"tail_ghost={self.tail_ghost_enabled}"
        )

    # =========================================================================
    # 主入口
    # =========================================================================

    def detect(self, angles: np.ndarray) -> List[DetectedRep]:
        if len(angles) < 10:
            return []

        n_frames = len(angles)
        min_distance = max(1, int(round(self.min_distance_sec * self.fps)))

        valleys, valley_props = find_peaks(
            -angles, prominence=self.min_prominence, distance=min_distance
        )
        peaks, peak_props = find_peaks(
            angles, prominence=self.min_prominence, distance=min_distance
        )

        if len(valleys) == 0:
            print(f"   ⚠️ 未检测到波谷 (valleys)")
            return []

        # ── Step 1: Multi-Candidate Generation (V7.1: 带 prominence) ──
        candidates = self._build_multi_candidates(
            angles, valleys, peaks, valley_props, peak_props
        )

        # ── Step 2: Biomechanical Scoring (V7.1: 多项加权) ──
        scored = self._score_candidates(candidates, angles, valleys, n_frames)

        # ── Step 3: DP Chain Selection (Shared Boundary Gate) ──
        final_chain = self._dp_chain_selection(scored)

        # ── Step 3.2: V7.2.5 Pause Split Merge ──
        final_chain, pause_merge_count = self._merge_pause_split_reps(final_chain, scored)

        # ── Step 3.6: V7.2.7 Internal Bridge Ghost Filter ──
        final_chain, bridge_count = self._filter_internal_bridge_ghosts(final_chain)

        # ── Step 3.8: V7.2.6 Tail Ghost Filter ──
        final_chain, tail_ghost_count = self._filter_tail_ghosts(final_chain)

        # ── Step 4: Rerack + Isolated Rest Filter ──
        final_reps, rerack_count = self._filter_rerack_reps(final_chain, n_frames)

        # ── 日志 ──
        self._print_v71_log(valleys, peaks, scored, final_reps, rerack_count,
                            pause_merge_count, tail_ghost_count, bridge_count)

        return final_reps

    # =========================================================================
    # Step 1: Multi-Candidate Generation (V7.1: 2×2, prominence, monotonicity pre-filter)
    # =========================================================================

    def _build_multi_candidates(
        self,
        angles: np.ndarray,
        valleys: np.ndarray,
        peaks: np.ndarray,
        valley_props: Dict,
        peak_props: Dict,
    ) -> List[DetectedRep]:
        """
        V7.1: 每个 valley 最多 2 prev_peaks × 2 next_peaks。
        同时记录 bottom_prominence、interior_valley_count。
        对每个组合做轻量级 monotonicity pre-filter，跳过明显跨周期的。
        """
        candidates: List[DetectedRep] = []
        sorted_peaks = np.sort(peaks)
        sorted_valleys = np.sort(valleys)
        min_phase_frames = max(int(self.min_eccentric_duration_s * self.fps), 5)

        # V7.1: 建立 prominence 映射
        valley_prom_map = {
            int(v): float(p)
            for v, p in zip(valleys, valley_props["prominences"])
        }

        for valley_idx in sorted_valleys:
            valley_idx = int(valley_idx)
            bottom_prom = valley_prom_map.get(valley_idx, 0.0)

            prev_peaks = sorted_peaks[sorted_peaks < valley_idx - min_phase_frames]
            next_peaks = sorted_peaks[sorted_peaks > valley_idx + min_phase_frames]

            if len(prev_peaks) == 0 or len(next_peaks) == 0:
                continue

            n_prev = min(self.max_candidates_per_valley, len(prev_peaks))
            n_next = min(self.max_candidates_per_valley, len(next_peaks))

            for pi in range(n_prev):
                start_peak = int(prev_peaks[-(pi + 1)])

                for ni in range(n_next):
                    end_peak = int(next_peaks[ni])

                    duration_frames = end_peak - start_peak
                    if duration_frames < int(self.absolute_min_duration_s * self.fps):
                        continue
                    if duration_frames > int(self.absolute_max_duration_s * self.fps):
                        continue

                    seg = angles[start_peak:end_peak + 1]
                    if len(seg) == 0:
                        continue

                    rom = float(np.nanmax(seg) - np.nanmin(seg))
                    if rom < self.absolute_min_rom_deg:
                        continue

                    # V7.1: 计算 interior valley count
                    interior_valleys = sorted_valleys[
                        (sorted_valleys > valley_idx + 3) &
                        (sorted_valleys < end_peak - 3)
                    ]
                    interior_count = len(interior_valleys)

                    # V7.1: 轻量级 monotonicity pre-filter
                    ecc_q, con_q, rev_pen = self._phase_direction_quality(
                        angles, start_peak, valley_idx, end_peak
                    )

                    # 如果方向质量极差（<0.4），直接跳过，减少无效候选
                    if ecc_q < 0.4 and con_q < 0.4:
                        continue

                    closure = self._windowed_closure(angles, start_peak, end_peak)
                    duration = duration_frames / self.fps
                    ecc_dur = (valley_idx - start_peak) / self.fps
                    con_dur = (end_peak - valley_idx) / self.fps

                    # V7.1: 计算 reversal counts
                    ecc_rev = self._count_reversals(angles[start_peak:valley_idx + 1], expected_sign=-1)
                    con_rev = self._count_reversals(angles[valley_idx:end_peak + 1], expected_sign=+1)

                    candidates.append(DetectedRep(
                        start_frame=start_peak,
                        bottom_frame=valley_idx,
                        end_frame=end_peak,
                        total_rom=round(rom, 1),
                        duration=round(duration, 2),
                        eccentric_duration=round(ecc_dur, 2),
                        concentric_duration=round(con_dur, 2),
                        closure=round(closure, 1),
                        bottom_prominence=round(bottom_prom, 1),
                        interior_valley_count=interior_count,
                        eccentric_reversal_count=ecc_rev,
                        concentric_reversal_count=con_rev,
                    ))

        candidates = self._deduplicate_candidates(candidates)
        return candidates

    # =========================================================================
    # Step 2: Biomechanical Scoring (V7.1: 多项加权)
    # =========================================================================

    def _score_candidates(
        self,
        candidates: List[DetectedRep],
        angles: np.ndarray,
        valleys: np.ndarray,
        n_frames: int,
    ) -> List[ScoredCandidate]:
        """
        V7.1: 多项加权评分。

        score = 0.20 * rom_score
              + 0.15 * duration_score
              + 0.20 * ecc_direction_score
              + 0.20 * con_direction_score
              + 0.10 * prominence_score
              + 0.10 * phase_balance_score
              + 0.05 * closure_score
              - interior_reversal_penalty
        """
        scored: List[ScoredCandidate] = []
        min_phase_frames = int(self.min_eccentric_duration_s * self.fps)

        for rep in candidates:
            sc = ScoredCandidate(rep=rep)
            sf, bf, ef = rep.start_frame, rep.bottom_frame, rep.end_frame

            # ── Hard reject: 仅极端情况 ──
            if rep.duration < self.absolute_min_duration_s:
                sc.hard_rejected = True
                sc.hard_reject_reason = "absolute_min_duration"
                scored.append(sc)
                continue
            if rep.total_rom < self.absolute_min_rom_deg:
                sc.hard_rejected = True
                sc.hard_reject_reason = "absolute_min_rom"
                scored.append(sc)
                continue
            if rep.duration > self.absolute_max_duration_s:
                sc.hard_rejected = True
                sc.hard_reject_reason = "absolute_max_duration"
                scored.append(sc)
                continue

            # Phase duration hard reject (V7.1: 0.35s / 0.30s)
            ecc_frames = bf - sf
            con_frames = ef - bf
            if ecc_frames < min_phase_frames:
                sc.hard_rejected = True
                sc.hard_reject_reason = "short_eccentric"
                scored.append(sc)
                continue
            if con_frames < min_phase_frames:
                sc.hard_rejected = True
                sc.hard_reject_reason = "short_concentric"
                scored.append(sc)
                continue

            # ── 分项评分 ──
            sub = {}

            # 1) ROM Score [0, 1]
            sub["rom"] = round(self._rom_score(rep.total_rom), 3)

            # 2) Duration Score [0, 1] (V7.1: 软限制)
            sub["duration"] = round(self._duration_score(rep.duration), 3)

            # 3-4) Phase Direction Quality (V7.1: 分阶段)
            ecc_q, con_q, reversal_penalty = self._phase_direction_quality(
                angles, sf, bf, ef
            )
            sub["ecc_direction"] = round(ecc_q, 3)
            sub["con_direction"] = round(con_q, 3)
            sub["reversal_penalty"] = round(reversal_penalty, 3)

            # 5) Bottom Prominence Score [0, 1] (V7.1: 新增)
            sub["prominence"] = round(self._prominence_score(rep.bottom_prominence), 3)

            # 6) Phase Balance Score [0, 1] (V7.1: 连续 log-ratio)
            sub["phase_balance"] = round(
                self._phase_balance_score(rep.eccentric_duration, rep.concentric_duration), 3
            )

            # 7) Closure Score [0, 1]
            sub["closure"] = round(self._closure_score(rep.closure), 3)

            # 8) Interior Valley Penalty (V7.1: 新增)
            interior_pen = self._interior_valley_penalty(angles, bf, ef, valleys)
            sub["interior_penalty"] = round(interior_pen, 3)

            # ── 加权总分 ──
            score = (
                0.20 * sub["rom"]
                + 0.15 * sub["duration"]
                + 0.18 * sub["ecc_direction"]
                + 0.18 * sub["con_direction"]
                + 0.10 * sub["prominence"]
                + 0.09 * sub["phase_balance"]
                + 0.10 * sub["closure"]
            )

            # 减去惩罚项
            score -= reversal_penalty
            score -= interior_pen

            # Clamp
            score = max(0.0, min(1.0, score))
            sc.score = round(score, 3)
            sc.sub_scores = sub

            if score < self.min_chain_score:
                sc.hard_rejected = True
                sc.hard_reject_reason = "low_score"

            scored.append(sc)

        return scored

    # =========================================================================
    # V7.1: 分项评分函数
    # =========================================================================

    def _rom_score(self, rom: float) -> float:
        """ROM 评分：18°~50° 线性上升，50°~130° 满分，>130° 轻微下降"""
        if rom < self.absolute_min_rom_deg:
            return 0.0
        if rom < self.ideal_min_rom_deg:
            return (rom - self.absolute_min_rom_deg) / (
                self.ideal_min_rom_deg - self.absolute_min_rom_deg
            )
        if rom <= self.ideal_max_rom_deg:
            return 1.0
        excess = (rom - self.ideal_max_rom_deg) / 30.0
        return max(0.6, 1.0 - excess * 0.2)

    def _duration_score(self, dur: float) -> float:
        """V7.1: Duration 软限制评分"""
        if dur <= 1.8:
            return 1.0
        elif dur <= 2.5:
            return 0.85
        elif dur <= 3.0:
            return 0.60
        else:
            return 0.15

    def _prominence_score(self, prom: float) -> float:
        """V7.1: Bottom prominence 评分"""
        if prom >= self.prominence_excellent:
            return 1.0
        elif prom >= self.prominence_good:
            return 0.85
        elif prom >= self.prominence_min:
            return 0.60
        else:
            return 0.30

    def _phase_balance_score(self, ecc_dur: float, con_dur: float) -> float:
        """V7.1: Phase balance 连续 log-ratio 评分"""
        if ecc_dur <= 0 or con_dur <= 0:
            return 0.3
        ratio = ecc_dur / con_dur
        ideal_ratio = 1.0
        deviation = abs(np.log(max(ratio, 1e-6) / ideal_ratio))
        penalty = min(0.5, deviation * 0.15)
        return max(0.0, 1.0 - penalty)

    def _closure_score(self, closure: float) -> float:
        """V7.2.5: 强化周期闭合度评分。closure<=10:1.0, <=20:0.85, <=30:0.60, <=45:0.25, >45:0.05"""
        if closure <= 10.0:
            return 1.0
        if closure <= 20.0:
            return 0.85
        if closure <= 30.0:
            return 0.60
        if closure <= 45.0:
            return 0.25
        return 0.05

    # =========================================================================
    # V7.1: Phase Direction Quality (核心修复)
    # =========================================================================

    def _phase_direction_quality(
        self, angles: np.ndarray, sf: int, bf: int, ef: int
    ) -> Tuple[float, float, float]:
        """
        V7.1: 分阶段检查方向单调性。

        返回:
          eccentric_quality: [0,1] 离心阶段角度下降的一致性
          concentric_quality: [0,1] 向心阶段角度上升的一致性
          reversal_penalty: concentric 中出现显著二次下降的惩罚
        """
        ecc_seg = angles[sf:bf + 1]
        con_seg = angles[bf:ef + 1]

        def _monotonic_quality(seg: np.ndarray, expected_sign: int) -> float:
            if len(seg) < 3:
                return 0.5
            d = np.diff(seg)
            valid = d[np.isfinite(d)]
            if len(valid) == 0:
                return 0.5
            correct = np.sum(valid * expected_sign > 0)
            return float(correct / len(valid))

        ecc_q = _monotonic_quality(ecc_seg, -1)  # 应该下降
        con_q = _monotonic_quality(con_seg, +1)  # 应该上升

        # V7.1: Concentric 中的二次下降检测
        reversal_penalty = 0.0
        if len(con_seg) >= 5:
            d = np.diff(con_seg)
            negative = d < 0
            runs = []
            run = 0
            for x in negative:
                if x:
                    run += 1
                else:
                    if run > 0:
                        runs.append(run)
                        run = 0
            if run > 0:
                runs.append(run)

            max_neg_run = max(runs, default=0)
            if max_neg_run >= 5:
                reversal_penalty = 0.30
            elif max_neg_run >= 4:
                reversal_penalty = 0.20
            elif max_neg_run >= 3:
                reversal_penalty = 0.10

        return ecc_q, con_q, reversal_penalty

    # =========================================================================
    # V7.1: Interior Valley Penalty
    # =========================================================================

    def _interior_valley_penalty(
        self, angles: np.ndarray, bf: int, ef: int, valleys: np.ndarray
    ) -> float:
        """
        V7.1: 检查 concentric 阶段中是否有显著的二次下降。
        基于 interior valley 相对于 bottom 的深度。
        """
        interior = [
            int(v) for v in valleys
            if bf + 3 < int(v) < ef - 3
        ]

        if not interior:
            return 0.0

        penalty = 0.0
        base = float(angles[bf])

        for v in interior:
            depth = float(angles[v]) - base
            if depth < -3.0:
                penalty += 0.20
            elif depth < 5.0:
                penalty += 0.10

        return min(0.40, penalty)

    # =========================================================================
    # V7.1: Reversal Counter
    # =========================================================================

    @staticmethod
    def _count_reversals(seg: np.ndarray, expected_sign: int, epsilon_deg: float = 1.5) -> int:
        """V7.2.5: 统计反向变化，带1.5° deadband，避免MediaPipe jitter被误判"""
        if len(seg) < 3:
            return 0
        d = np.diff(seg)
        valid = d[np.isfinite(d)]
        if len(valid) == 0:
            return 0
        significant = valid[np.abs(valid) >= epsilon_deg]
        if len(significant) == 0:
            return 0
        wrong = np.sum(significant * expected_sign < 0)
        return int(wrong)

    # =========================================================================
    # Step 3.1: V7.2.4 Shared Boundary Gate
    # =========================================================================

    def _shared_boundary_transition_allowed(
        self, prev: ScoredCandidate, curr: ScoredCandidate
    ) -> bool:
        """
        V7.2.4: shared boundary (prev.end == curr.start) 不能无条件放行。
        强候选(score>=0.60)或低分但强底部(prom>=60, closure<=20, con_rev<=25)才允许。
        """
        r = curr.rep
        if prev.rep.end_frame != r.start_frame:
            return True
        if curr.score >= 0.60:
            return True
        if (curr.score >= 0.45 and r.bottom_prominence >= 60.0
                and r.closure <= 20.0 and r.concentric_reversal_count <= 25):
            return True
        return False

    # =========================================================================
    # Step 3: DP Chain Selection (V7.1: cadence + selection cost, NO shared boundary)
    # =========================================================================

    def _dp_chain_selection(
        self, scored: List[ScoredCandidate]
    ) -> List[DetectedRep]:
        """
        V7.1 DP:
          - rep_selection_cost: 每选一个 Rep 扣固定成本，消除 Cardinality Bias
          - Cadence transition: 基于 bottom-to-bottom 间隔的节奏一致性
          - 不做 shared_boundary_protection，overlap 由 DP 约束自然处理
        """
        valid = [s for s in scored if not s.hard_rejected]

        if len(valid) == 0:
            return self._single_rep_recovery(scored)

        valid.sort(key=lambda s: s.rep.start_frame)
        n = len(valid)

        # V7.1: 估算目标周期（用于 cadence）
        target_cycle = self._estimate_target_cycle(valid)

        dp = [0.0] * n
        parent = [-1] * n

        for i in range(n):
            # V7.1: 初始化 = 自身分数 - 选择成本
            dp[i] = valid[i].score - self.rep_selection_cost
            parent[i] = -1

            for j in range(i):
                # V7.2.4: Overlap + Shared Boundary Gate
                prev_end = valid[j].rep.end_frame
                curr_start = valid[i].rep.start_frame
                if prev_end > curr_start:
                    continue
                if prev_end == curr_start:
                    if not self._shared_boundary_transition_allowed(valid[j], valid[i]):
                        continue

                # Bottom spacing 检查
                bottom_gap = valid[i].rep.bottom_frame - valid[j].rep.bottom_frame
                if bottom_gap < self.min_bottom_to_bottom_frames:
                    continue

                # V7.1: Cadence bonus
                cadence_bonus = self._cadence_bonus(
                    valid[j], valid[i], target_cycle
                )

                # V7.1: transition = 前序总分 + 当前分数 + 节奏奖励 - 选择成本
                transition_score = (
                    dp[j]
                    + valid[i].score
                    + cadence_bonus
                    - self.rep_selection_cost
                )

                if transition_score > dp[i]:
                    dp[i] = transition_score
                    parent[i] = j

        # 找最优结尾
        best_end = int(np.argmax(dp))
        if dp[best_end] < 0:
            return self._single_rep_recovery(scored)

        # 回溯链
        chain_indices = []
        idx = best_end
        while idx >= 0:
            chain_indices.append(idx)
            idx = parent[idx]
        chain_indices.reverse()

        final_reps = []
        for ci in chain_indices:
            rep = valid[ci].rep
            rep.score = valid[ci].score
            valid[ci].in_final_chain = True
            final_reps.append(rep)

        return final_reps

    def _estimate_target_cycle(self, valid: List[ScoredCandidate]) -> float:
        """V7.2.4: 从 bottom-to-bottom gap 估算目标周期"""
        bottoms = sorted({c.rep.bottom_frame for c in valid if c.rep.bottom_frame >= 0})
        if len(bottoms) >= 4:
            gaps = np.diff(bottoms) / self.fps
            gaps = gaps[(gaps >= 0.8) & (gaps <= 4.0)]
            if len(gaps) >= 3:
                return float(np.median(gaps))
        return self.target_cycle_duration_s

    def _cadence_bonus(
        self,
        prev: ScoredCandidate,
        curr: ScoredCandidate,
        target_cycle_s: float,
    ) -> float:
        """V7.2.4: Bottom-to-Bottom cadence，降低幅度"""
        gap_s = (curr.rep.bottom_frame - prev.rep.bottom_frame) / self.fps

        if gap_s <= 0:
            return -0.20

        ratio = gap_s / max(target_cycle_s, 0.5)

        if 0.85 <= ratio <= 1.15:
            return 0.08
        elif 0.70 <= ratio <= 1.30:
            return 0.03
        elif 0.55 <= ratio <= 1.50:
            return -0.03
        else:
            return -0.10


    # =========================================================================
    # =========================================================================
    # Step 3.6: V7.2.7 Internal Bridge Ghost Filter（弱桥接候选过滤）
    # =========================================================================

    def _filter_internal_bridge_ghosts(
        self, reps: List[DetectedRep]
    ) -> Tuple[List[DetectedRep], int]:
        """
        V7.2.7: 过滤"真实Rep → 弱桥接候选 → 真实Rep"中的弱候选。

        与 Tail Ghost 区别：Tail Ghost 只检查最后一个 Rep；此过滤器检查所有位置。
        与 Pause Merge 区别：Pause Merge 合并两个 closure 都差的半周期；此过滤器删除一个弱的桥接候选。

        条件（情况A：当前弱，下一个强）：
        - boundary_gap <= bridge_boundary_frames
        - current.score <= bridge_max_score AND current.prom <= bridge_max_prominence
        - next.score >= bridge_min_strong_score
        - next.score - current.score >= bridge_min_score_gap

        条件（情况B：上一个强，当前弱）：
        - boundary_gap <= bridge_boundary_frames
        - current.score <= bridge_max_score AND current.prom <= bridge_max_prominence
        - prev.score >= bridge_min_strong_score
        - prev.score - current.score >= bridge_min_score_gap
        """
        if not self.bridge_enabled or len(reps) < 2:
            return reps, 0

        kept: List[DetectedRep] = []
        filtered = 0

        for i, cur in enumerate(reps):
            remove_current = False

            # 情况A：当前弱，下一个强
            if i < len(reps) - 1:
                nxt = reps[i + 1]
                boundary_gap = abs(cur.end_frame - nxt.start_frame)
                score_gap = nxt.score - cur.score
                current_weak = (cur.score <= self.bridge_max_score
                                and cur.bottom_prominence <= self.bridge_max_prominence)
                next_strong = nxt.score >= self.bridge_min_strong_score
                if (boundary_gap <= self.bridge_boundary_frames
                        and current_weak and next_strong
                        and score_gap >= self.bridge_min_score_gap):
                    print(
                        f"   👻 INTERNAL-BRIDGE filtered: "
                        f"[{cur.start_frame}-{cur.bottom_frame}-{cur.end_frame}] "
                        f"→ next [{nxt.start_frame}-{nxt.bottom_frame}-{nxt.end_frame}] "
                        f"score={cur.score:.3f}->{nxt.score:.3f} "
                        f"prom={cur.bottom_prominence:.0f}->{nxt.bottom_prominence:.0f} "
                        f"boundary={boundary_gap}"
                    )
                    remove_current = True

            # 情况B：上一个强，当前弱
            if not remove_current and i > 0:
                prev = reps[i - 1]
                boundary_gap = abs(prev.end_frame - cur.start_frame)
                score_gap = prev.score - cur.score
                current_weak = (cur.score <= self.bridge_max_score
                                and cur.bottom_prominence <= self.bridge_max_prominence)
                prev_strong = prev.score >= self.bridge_min_strong_score
                if (boundary_gap <= self.bridge_boundary_frames
                        and current_weak and prev_strong
                        and score_gap >= self.bridge_min_score_gap):
                    print(
                        f"   👻 INTERNAL-BRIDGE filtered: "
                        f"prev [{prev.start_frame}-{prev.bottom_frame}-{prev.end_frame}] "
                        f"→ [{cur.start_frame}-{cur.bottom_frame}-{cur.end_frame}] "
                        f"score={prev.score:.3f}->{cur.score:.3f} "
                        f"prom={prev.bottom_prominence:.0f}->{cur.bottom_prominence:.0f} "
                        f"boundary={boundary_gap}"
                    )
                    remove_current = True

            if remove_current:
                filtered += 1
            else:
                kept.append(cur)

        return kept, filtered

    # =========================================================================
    # Step 3.8: V7.2.6 Tail Ghost Filter（尾部伪 rep 过滤）
    # =========================================================================

    def _filter_tail_ghosts(
        self, reps: List[DetectedRep]
    ) -> Tuple[List[DetectedRep], int]:
        """
        V7.2.6: 过滤 set 尾部的伪 Rep（回架/摆位/身体调整被误识别）。

        只检查最后一个 Rep，五个联合条件：
        1. 上一个 Rep 强 (score >= 0.75)
        2. 最后一个 Rep 弱 (score <= 0.65)
        3. 间隔短 (gap <= 1.2s)
        4. ROM 明显下降 (ratio <= 0.72)
        5. prominence 明显下降 (ratio <= 0.55)
        """
        if not self.tail_ghost_enabled:
            return reps, 0
        if len(reps) < 2:
            return reps, 0

        prev = reps[-2]
        last = reps[-1]

        # 1. 上一个必须强
        if prev.score < self.tail_ghost_min_strong_score:
            return reps, 0
        # 2. 最后一个必须弱
        if last.score > self.tail_ghost_max_weak_score:
            return reps, 0
        # 3. 间隔短
        gap_sec = (last.start_frame - prev.end_frame) / self.fps
        if gap_sec < 0 or gap_sec > self.tail_ghost_max_gap_sec:
            return reps, 0
        # 4. ROM 明显下降
        rom_ratio = last.total_rom / max(prev.total_rom, 1e-6)
        if rom_ratio > self.tail_ghost_rom_ratio:
            return reps, 0
        # 5. prominence 明显下降
        prom_ratio = last.bottom_prominence / max(prev.bottom_prominence, 1e-6)
        if prom_ratio > self.tail_ghost_prom_ratio:
            return reps, 0

        print(
            f"   👻 TAIL GHOST filtered: "
            f"[{last.start_frame}-{last.bottom_frame}-{last.end_frame}] "
            f"after [{prev.start_frame}-{prev.bottom_frame}-{prev.end_frame}] "
            f"gap={gap_sec:.2f}s "
            f"score={prev.score:.3f}->{last.score:.3f} "
            f"ROM={prev.total_rom:.1f}->{last.total_rom:.1f}({rom_ratio:.2f}) "
            f"prom={prev.bottom_prominence:.0f}->{last.bottom_prominence:.0f}({prom_ratio:.2f})"
        )

        return reps[:-1], 1

    # =========================================================================
    # Step 3.2: V7.2.5 Pause Split Merge（暂停卧推半周期合并）
    # =========================================================================

    def _merge_pause_split_reps(
        self, final_chain: List[DetectedRep], scored: List[ScoredCandidate]
    ) -> Tuple[List[DetectedRep], int]:
        """
        V7.2.5: 合并暂停卧推被拆成的两个半周期候选。
        两个共享边界、closure 都 >=40° 的候选，若存在完整候选 (closure<=15°) 则合并。
        """
        if len(final_chain) < 2:
            return final_chain, 0

        candidate_map: Dict[Tuple[int, int, int], ScoredCandidate] = {}
        for sc in scored:
            r = sc.rep
            candidate_map[(int(r.start_frame), int(r.bottom_frame), int(r.end_frame))] = sc

        merged_chain: List[DetectedRep] = []
        merge_count = 0
        i = 0
        while i < len(final_chain):
            if i >= len(final_chain) - 1:
                merged_chain.append(final_chain[i])
                break
            prev = final_chain[i]
            curr = final_chain[i + 1]
            # 共享边界
            if prev.end_frame != curr.start_frame:
                merged_chain.append(prev)
                i += 1
                continue
            # 两个 closure 都很差
            if prev.closure < 40.0 or curr.closure < 40.0:
                merged_chain.append(prev)
                i += 1
                continue
            # 存在完整候选
            merged_key = (int(prev.start_frame), int(prev.bottom_frame), int(curr.end_frame))
            merged_sc = candidate_map.get(merged_key)
            if merged_sc is None:
                merged_chain.append(prev)
                i += 1
                continue
            merged = merged_sc.rep
            # ROM 足够
            min_rom = max(55.0, 0.80 * max(prev.total_rom, curr.total_rom))
            if merged.total_rom < min_rom or merged.duration > 3.50 or merged.closure > 15.0:
                merged_chain.append(prev)
                i += 1
                continue
            # 两个 bottom 间距足够
            if (curr.bottom_frame - prev.bottom_frame) < int(0.25 * self.fps):
                merged_chain.append(prev)
                i += 1
                continue
            print(
                f"   🔗 PAUSE MERGE: "
                f"[{prev.start_frame}-{prev.bottom_frame}-{prev.end_frame}] "
                f"+ [{curr.start_frame}-{curr.bottom_frame}-{curr.end_frame}] "
                f"→ [{merged.start_frame}-{merged.bottom_frame}-{merged.end_frame}] "
                f"ROM={merged.total_rom:.1f}° closure={merged.closure:.1f}° "
                f"local=({prev.closure:.1f},{curr.closure:.1f})"
            )
            merged.score = max(prev.score, curr.score, merged.score)
            merged_chain.append(merged)
            merge_count += 1
            i += 2
        return merged_chain, merge_count

    # =========================================================================
    # Step 4: Rerack Filter
    # =========================================================================

    def _filter_rerack_reps(
        self, reps: List[DetectedRep], n_frames: int
    ) -> Tuple[List[DetectedRep], int]:
        if len(reps) < 2:
            return reps, 0

        idle_gaps = [
            (reps[i + 1].start_frame - reps[i].end_frame) / self.fps
            for i in range(len(reps) - 1)
        ]
        median_gap = float(np.median(idle_gaps)) if idle_gaps else 0.0
        median_rom = float(np.median([r.total_rom for r in reps]))
        gap_threshold = max(self.rerack_min_gap_s, self.rerack_gap_multiplier * median_gap)

        kept: List[DetectedRep] = []
        filtered = 0
        for i, rep in enumerate(reps):
            prev_gap = (rep.start_frame - reps[i - 1].end_frame) / self.fps if i > 0 else None
            next_gap = (reps[i + 1].start_frame - rep.end_frame) / self.fps if i < len(reps) - 1 else None
            neighbor_gaps = [g for g in (prev_gap, next_gap) if g is not None]
            neighbor_gap = min(neighbor_gaps) if neighbor_gaps else 0

            near_edge = (rep.start_frame < self.rerack_edge_frames or
                         rep.end_frame > n_frames - self.rerack_edge_frames)
            low_rom = rep.total_rom < self.rerack_rom_ratio * median_rom
            long_next_gap = (next_gap is not None and next_gap > self.rerack_long_next_gap_s)

            # V7.2.5: Isolated Rest Filter - 长时间rest后出现的孤立候选，closure差
            isolated_after_rest = (
                neighbor_gap >= 2.0 and rep.closure >= 35.0 and not near_edge
            )

            if isolated_after_rest:
                print(
                    f"   🔚 ISOLATED-REST filtered: "
                    f"[{rep.start_frame}-{rep.bottom_frame}-{rep.end_frame}] "
                    f"gap={neighbor_gap:.2f}s closure={rep.closure:.1f}°"
                )
                filtered += 1
            elif (neighbor_gap > gap_threshold and near_edge and low_rom) or (long_next_gap and low_rom):
                filtered += 1
            else:
                kept.append(rep)

        return kept, filtered

    # =========================================================================
    # Single-Rep Recovery
    # =========================================================================

    def _single_rep_recovery(self, scored: List[ScoredCandidate]) -> List[DetectedRep]:
        best = None
        best_score = -1.0
        for sc in scored:
            rep = sc.rep
            if rep.total_rom < self.absolute_min_rom_deg:
                continue
            if rep.duration < self.absolute_min_duration_s:
                continue
            if rep.duration > self.absolute_max_duration_s:
                continue
            s = min(1.0, rep.total_rom / self.ideal_min_rom_deg)
            if s > best_score:
                best_score = s
                best = rep
        if best is not None and best_score > 0.3:
            best.score = best_score
            return [best]
        return []

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def _deduplicate_candidates(self, candidates: List[DetectedRep]) -> List[DetectedRep]:
        if not candidates:
            return []
        best_by_key: Dict[tuple, DetectedRep] = {}
        for rep in candidates:
            key = (rep.start_frame, rep.bottom_frame, rep.end_frame)
            if key not in best_by_key or rep.total_rom > best_by_key[key].total_rom:
                best_by_key[key] = rep
        return sorted(best_by_key.values(), key=lambda r: r.start_frame)

    def _windowed_closure(self, angles: np.ndarray, sf: int, ef: int) -> float:
        w = 3
        n = len(angles)
        s_lo, s_hi = max(0, sf - w), min(n, sf + w + 1)
        e_lo, e_hi = max(0, ef - w), min(n, ef + w + 1)
        s_mean = float(np.nanmean(angles[s_lo:s_hi])) if s_hi > s_lo else float(angles[sf])
        e_mean = float(np.nanmean(angles[e_lo:e_hi])) if e_hi > e_lo else float(angles[ef])
        return abs(s_mean - e_mean)

    # =========================================================================
    # V7.1 日志
    # =========================================================================

    def _print_v71_log(
        self, valleys, peaks,
        scored: List[ScoredCandidate],
        final_reps: List[DetectedRep],
        rerack_count: int,
        pause_merge_count: int = 0,
        tail_ghost_count: int = 0,
        bridge_count: int = 0,
    ):
        print(f"\n📊 V8 CycleRepDetector (v7.2.7) Biomechanical Candidate Graph:")
        print(f"   Extrema: {len(valleys)} valleys, {len(peaks)} peaks")
        print(f"   Total candidates: {len(scored)}")
        print()

        print(f"   ┌─ Candidate Scoring (multi-weighted) ─────────────────────")
        for sc in scored:
            r = sc.rep
            if sc.hard_rejected:
                status = f"❌ HARD ({sc.hard_reject_reason})"
            elif sc.in_final_chain:
                status = f"✅ FINAL (score={sc.score:.3f})"
            else:
                status = f"⬜ skip (score={sc.score:.3f})"

            print(
                f"   │ [{r.start_frame:>4d}-{r.bottom_frame:>4d}-{r.end_frame:>4d}]  "
                f"ROM={r.total_rom:>5.1f}°  dur={r.duration:.2f}s  "
                f"ecc={r.eccentric_duration:.2f}s  con={r.concentric_duration:.2f}s  "
                f"closure={r.closure:.1f}°  prom={r.bottom_prominence:.0f}"
            )

            if sc.sub_scores:
                parts = []
                for k, v in sc.sub_scores.items():
                    parts.append(f"{k}={v:.2f}")
                print(f"   │   sub: {', '.join(parts)}")

            print(
                f"   │   meta: int_valleys={r.interior_valley_count}  "
                f"ecc_rev={r.eccentric_reversal_count}  "
                f"con_rev={r.concentric_reversal_count}"
            )

            print(f"   │   → {status}")

        print(f"   └────────────────────────────────────────────────────────────")

        if rerack_count > 0:
            print(f"   🔚 Rerack filtered: {rerack_count}")

        if pause_merge_count > 0:
            print(f"   🔗 Pause merge (split half-cycles): {pause_merge_count}")

        if bridge_count > 0:
            print(f"   👻 Internal bridge filtered (weak bridge candidate): {bridge_count}")

        if tail_ghost_count > 0:
            print(f"   👻 Tail ghost filtered (post-rep oscillation): {tail_ghost_count}")

        if rerack_count > 0:
            print(f"   🔚 Rerack filtered: {rerack_count}")

        print(f"\n   ✅ Final validated reps: {len(final_reps)}")
        for i, r in enumerate(final_reps):
            print(
                f"      Rep {i+1}: [{r.start_frame}-{r.bottom_frame}-{r.end_frame}] "
                f"ROM={r.total_rom:.1f}° dur={r.duration:.2f}s "
                f"score={r.score:.3f} prom={r.bottom_prominence:.0f}"
            )


# ═══════════════════════════════════════════════════════
# PhaseBuilder
# ═══════════════════════════════════════════════════════

class PhaseBuilder:
    """将 CycleRepDetector 的事件锚点转化为完整的 RepContext。"""

    def __init__(self, fps: float, min_rom_deg: float = 40.0,
                 reject_rom_deg: float = 15.0):
        self.fps = fps
        self.min_rom_deg = min_rom_deg
        self.reject_rom_deg = reject_rom_deg

    # ═══════════════════════════════════════
    #  V3.1 Bounce 特征辅助方法
    # ═══════════════════════════════════════

    def _compute_bottom_dwell(self, angles: np.ndarray, bottom_idx: int) -> float:
        """
        V3.1: 基于 near-zero velocity 计算真实底部停留时间。
        不再是固定窗口长度。

        从 bottom_idx 向左右扩展，直到 |velocity| > 20°/s。
        """
        if len(angles) < 5 or bottom_idx <= 0 or bottom_idx >= len(angles) - 1:
            return 0.0

        vel = np.gradient(angles, 1.0 / self.fps)
        PAUSE_VEL_THRESH = 20.0  # °/s

        # 向左扩展
        left = bottom_idx
        while left > 0 and abs(vel[left - 1]) <= PAUSE_VEL_THRESH:
            left -= 1

        # 向右扩展
        right = bottom_idx
        while right < len(vel) - 1 and abs(vel[right + 1]) <= PAUSE_VEL_THRESH:
            right += 1

        return (right - left) / self.fps

    def _compute_reversal_frames(self, angles: np.ndarray, bottom_idx: int) -> int:
        """
        V3.1: 真正计算方向反转用了多少帧。
        找底部前最后一个明显下降帧 → 底部后第一个明显上升帧。

        返回帧数间隔，999 表示无法计算。
        """
        if bottom_idx <= 0 or bottom_idx >= len(angles) - 1:
            return 999

        vel = np.gradient(angles, 1.0 / self.fps)

        # 底部前最后一个明显下降帧（vel < -10）
        pre_idx = None
        for i in range(bottom_idx - 1, max(-1, bottom_idx - 8), -1):
            if vel[i] < -10.0:
                pre_idx = i
                break

        # 底部后第一个明显上升帧（vel > 10）
        post_idx = None
        for i in range(bottom_idx + 1, min(len(vel), bottom_idx + 10)):
            if vel[i] > 10.0:
                post_idx = i
                break

        if pre_idx is None or post_idx is None:
            return 999

        return max(0, post_idx - pre_idx - 1)

    def build(
        self,
        rep_index: int,
        start_frame: int,
        bottom_frame: int,
        end_frame: int,
        angles: np.ndarray,
        left_elbow: Optional[np.ndarray] = None,
        right_elbow: Optional[np.ndarray] = None,
        bilateral_valid_ratio: float = 1.0,
        signal_source: str = "bilateral",
        left_wrist: Optional[np.ndarray] = None,
        right_wrist: Optional[np.ndarray] = None,
        left_upper_arm_torso: Optional[np.ndarray] = None,
        right_upper_arm_torso: Optional[np.ndarray] = None,
        left_hip: Optional[np.ndarray] = None,
        right_hip: Optional[np.ndarray] = None,
        left_shoulder: Optional[np.ndarray] = None,
        right_shoulder: Optional[np.ndarray] = None,
        left_knee: Optional[np.ndarray] = None,
        right_knee: Optional[np.ndarray] = None,
    ) -> RepContext:
        n = len(angles)
        sf = max(0, min(start_frame, n - 1))
        bf = max(sf, min(bottom_frame, n - 1))
        ef = max(bf, min(end_frame, n - 1))

        seg = angles[sf:ef + 1]
        if len(seg) > 0:
            actual_max = float(np.nanmax(seg))
            actual_min = float(np.nanmin(seg))
        else:
            actual_max = actual_min = float(angles[sf])
        actual_rom = actual_max - actual_min

        top_angle = actual_max
        bottom_angle = actual_min
        top2_angle = actual_max

        con_seg_data = angles[bf:ef + 1] if bf < ef else seg
        if len(con_seg_data) > 0:
            top2_angle = float(np.nanmax(con_seg_data))

        start_time = sf / self.fps
        bottom_time = bf / self.fps
        end_time = ef / self.fps
        total_duration = end_time - start_time

        status = ValidationStatus.VALID
        if actual_rom < self.reject_rom_deg:
            status = ValidationStatus.REJECTED_ROM
        elif actual_rom < self.min_rom_deg:
            status = ValidationStatus.SHORT_ROM
        elif total_duration <= 0:
            status = ValidationStatus.INVALID_DURATION

        ecc_start, ecc_end = sf, bf
        con_start, con_end = bf, ef
        # bot_zone 用于 phases 展示，不用于 dwell 计算
        bot_zone_start = max(sf, bf - int(0.1 * self.fps))
        bot_zone_end = min(ef, bf + int(0.1 * self.fps))

        ecc_dur = (ecc_end - ecc_start) / self.fps if ecc_end > ecc_start else 0.0
        con_dur = (con_end - con_start) / self.fps if con_end > con_start else 0.0
        # V3.1: 真正的 bottom dwell，基于 near-zero velocity，不再是固定窗口长度
        bot_dwell = self._compute_bottom_dwell(angles, bf)

        ecc_vel = con_vel = None
        peak_ecc = peak_con = mean_con = None

        if ecc_end > ecc_start and ecc_dur > 0:
            ecc_seg = angles[ecc_start:ecc_end + 1]
            ecc_vel = np.gradient(ecc_seg, 1.0 / self.fps)
            peak_ecc = float(np.nanmin(ecc_vel))

        if con_end > con_start and con_dur > 0:
            con_seg = angles[con_start:con_end + 1]
            con_vel = np.gradient(con_seg, 1.0 / self.fps)
            peak_con = float(np.nanmax(con_vel))
            mean_con = float(np.nanmean(con_vel))

        # V3.1: 修复 bounce 特征的物理语义
        # 1. pre_bottom_velocity: 卧推下放时 elbow angle 下降，velocity < 0
        #    取负号使快速下放为正值，用 median 抗 jitter
        # 2. bottom_acceleration: 真正的二阶导数 d(velocity)/dt，单位 °/s²
        # 3. direction_reversal_frames: 真正计算反转用了多少帧，不再是 0/1
        pre_bot_vel = bot_accel = None
        dir_reversal = 999  # 默认值，表示无法计算
        if bf > 0 and bf < n - 1:
            window = max(2, int(0.05 * self.fps))
            pre_seg = angles[max(0, bf - window):bf]
            post_seg = angles[bf:min(n, bf + window + 1)]

            if len(pre_seg) > 1:
                pre_vel_series = np.gradient(pre_seg, 1.0 / self.fps)
                # 取负号：下放时 velocity < 0，快速下放 → pre_bottom_velocity > 0
                pre_bot_vel = float(-np.nanmedian(pre_vel_series))

            if len(post_seg) > 1:
                post_vel = np.gradient(post_seg, 1.0 / self.fps)
                # 真正的加速度 = d(velocity)/dt
                post_accel = np.gradient(post_vel, 1.0 / self.fps)
                bot_accel = float(np.nanmedian(post_accel))

            # 真正计算方向反转帧数
            dir_reversal = self._compute_reversal_frames(angles, bf)

        le_slice = right_elbow_slice = bilat_slice = None
        left_wrist_slice = right_wrist_slice = None
        src = SignalSource.BILATERAL

        if left_elbow is not None and sf < len(left_elbow):
            le_slice = left_elbow[sf:ef + 1].copy()
        if right_elbow is not None and sf < len(right_elbow):
            right_elbow_slice = right_elbow[sf:ef + 1].copy()

        # wrist 坐标切片（形状 (N, 2)）
        if left_wrist is not None and sf < len(left_wrist):
            left_wrist_slice = left_wrist[sf:ef + 1].copy()
        if right_wrist is not None and sf < len(right_wrist):
            right_wrist_slice = right_wrist[sf:ef + 1].copy()

        # upper_arm_torso 角度切片
        left_uat_slice = right_uat_slice = None
        if left_upper_arm_torso is not None and sf < len(left_upper_arm_torso):
            left_uat_slice = left_upper_arm_torso[sf:ef + 1].copy()
        if right_upper_arm_torso is not None and sf < len(right_upper_arm_torso):
            right_uat_slice = right_upper_arm_torso[sf:ef + 1].copy()

        # 髋/肩/膝坐标切片（形状 (N, 2)，用于 butt-off-bench 检测）
        left_hip_slice = right_hip_slice = None
        left_shoulder_slice = right_shoulder_slice = None
        left_knee_slice = right_knee_slice = None

        if left_hip is not None and sf < len(left_hip):
            left_hip_slice = left_hip[sf:ef + 1].copy()
        if right_hip is not None and sf < len(right_hip):
            right_hip_slice = right_hip[sf:ef + 1].copy()
        if left_shoulder is not None and sf < len(left_shoulder):
            left_shoulder_slice = left_shoulder[sf:ef + 1].copy()
        if right_shoulder is not None and sf < len(right_shoulder):
            right_shoulder_slice = right_shoulder[sf:ef + 1].copy()
        if left_knee is not None and sf < len(left_knee):
            left_knee_slice = left_knee[sf:ef + 1].copy()
        if right_knee is not None and sf < len(right_knee):
            right_knee_slice = right_knee[sf:ef + 1].copy()

        if le_slice is not None and right_elbow_slice is not None:
            le_valid = np.isfinite(le_slice)
            re_valid = np.isfinite(right_elbow_slice)
            both_valid = le_valid & re_valid
            if both_valid.any():
                bilat = np.full_like(le_slice, np.nan)
                bilat[both_valid] = (le_slice[both_valid] + right_elbow_slice[both_valid]) / 2.0
                bilat_slice = bilat
                src = SignalSource.BILATERAL
            elif le_valid.any() and not re_valid.any():
                bilat_slice = le_slice
                src = SignalSource.LEFT_ONLY
            elif re_valid.any() and not le_valid.any():
                bilat_slice = right_elbow_slice
                src = SignalSource.RIGHT_ONLY
            else:
                bilat_slice = None
                src = SignalSource.BILATERAL
        elif le_slice is not None:
            bilat_slice = le_slice
            src = SignalSource.LEFT_ONLY
        elif right_elbow_slice is not None:
            bilat_slice = right_elbow_slice
            src = SignalSource.RIGHT_ONLY

        if signal_source != "bilateral":
            try:
                src = SignalSource(signal_source)
            except ValueError:
                pass

        phases: List[PhaseSegment] = []
        if ecc_end > ecc_start:
            phases.append(PhaseSegment(
                name="eccentric",
                start_frame=ecc_start, end_frame=ecc_end,
                start_time=ecc_start / self.fps, end_time=ecc_end / self.fps,
                peak_velocity=peak_ecc,
                mean_velocity=float(np.nanmean(ecc_vel)) if ecc_vel is not None else None,
            ))
        if bot_zone_end > bot_zone_start:
            phases.append(PhaseSegment(
                name="bottom", 
                start_frame=bot_zone_start, end_frame=bot_zone_end,
                start_time=bot_zone_start / self.fps, end_time=bot_zone_end / self.fps,
            ))
        if con_end > con_start:
            phases.append(PhaseSegment(
                name="concentric",
                start_frame=con_start, end_frame=con_end,
                start_time=con_start / self.fps, end_time=con_end / self.fps,
                peak_velocity=peak_con, mean_velocity=mean_con,
            ))

        return RepContext(
            rep_index=rep_index,
            start_frame=sf, bottom_frame=bf, end_frame=ef,
            top_angle=top_angle, bottom_angle=bottom_angle, top2_angle=top2_angle,
            actual_max_angle=actual_max, actual_min_angle=actual_min,
            actual_rom=actual_rom,
            start_time=start_time, bottom_time=bottom_time, end_time=end_time,
            total_duration=total_duration,
            eccentric_start=ecc_start, eccentric_end=ecc_end,
            concentric_start=con_start, concentric_end=con_end,
            bottom_zone_start=bot_zone_start, bottom_zone_end=bot_zone_end,
            eccentric_duration=ecc_dur, concentric_duration=con_dur,
            bottom_dwell_time=bot_dwell,
            eccentric_velocity=ecc_vel, concentric_velocity=con_vel,
            peak_eccentric_velocity=peak_ecc, peak_concentric_velocity=peak_con,
            mean_concentric_velocity=mean_con,
            pre_bottom_velocity=pre_bot_vel, bottom_acceleration=bot_accel,
            direction_reversal_frames=dir_reversal,
            left_elbow=le_slice, right_elbow=right_elbow_slice,
            bilateral_elbow=bilat_slice,
            bilateral_valid_ratio=bilateral_valid_ratio,
            signal_source=src, phases=phases,
            validation_status=status, fps=self.fps,
            left_wrist=left_wrist_slice, right_wrist=right_wrist_slice,
            left_upper_arm_torso=left_uat_slice,
            right_upper_arm_torso=right_uat_slice,
            left_hip=left_hip_slice,
            right_hip=right_hip_slice,
            left_shoulder=left_shoulder_slice,
            right_shoulder=right_shoulder_slice,
            left_knee=left_knee_slice,
            right_knee=right_knee_slice,
        )