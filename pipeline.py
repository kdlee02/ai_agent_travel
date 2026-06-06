"""
pipeline.py

SeoulMate AI 여행 플래너 파이프라인.

Generator(plan_node)가 만든 초안 itinerary를 받아서:
1. critic    → 문제 감지
2. replanner → 구조적 문제 있는 day 재구성
3. critic    → 재검수
4. repair    → 작은 이슈 수정
5. critic    → 최종 검수
→ passed itinerary 반환

사용법
------
# 코드에서 직접 호출
from pipeline import run_pipeline
final_itinerary, pipeline_log = run_pipeline(raw_itinerary, user_state)

# CLI 테스트
python pipeline.py --input output/replanned_itinerary_v3_1.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


# ============================================================
# 파이프라인 설정
# ============================================================

MAX_REPLAN_CYCLES = 2   # replanner 최대 반복 횟수 (무한루프 방지)
MAX_REPAIR_CYCLES = 1   # repair 최대 반복 횟수
PROFILE_PATH = "output/area_profiles_v2.json"


# ============================================================
# 파이프라인 실행
# ============================================================

def run_pipeline(
    itinerary: dict[str, Any],
    user_state: dict[str, Any] | None = None,
    profile_path: str | Path | None = None,
    verbose: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    초안 itinerary를 받아 critic → replanner → repair 파이프라인을 실행.

    Parameters
    ----------
    itinerary   : Generator(plan_node)가 만든 초안 itinerary dict
    user_state  : 사용자 정보 (purpose, location, duration 등)
    profile_path: area_profiles_v2.json 경로 (기본값 사용 가능)
    verbose     : True면 각 단계 로그 출력

    Returns
    -------
    (final_itinerary, pipeline_log)
    - final_itinerary : 파이프라인 통과한 최종 itinerary
    - pipeline_log    : 각 단계 결과 요약
    """
    from critic import evaluate_itinerary
    from replanner import replan_itinerary
    from repair import repair_itinerary

    user_state = user_state or {}
    base_dir = Path(__file__).resolve().parent
    if profile_path is None:
        profile_path = base_dir / PROFILE_PATH
    else:
        profile_path = Path(profile_path)
        if not profile_path.is_absolute():
            profile_path = base_dir / profile_path

    current = copy.deepcopy(itinerary)
    log: dict[str, Any] = {
        "cycles": [],
        "final_score": None,
        "final_passed": False,
        "total_changes": 0,
    }

    def _log(msg: str) -> None:
        if verbose:
            print(f"[pipeline] {msg}")

    # ------------------------------------------------
    # Step 1: 초기 critic
    # ------------------------------------------------
    _log("Step 1 — 초기 critic 실행")
    critic_result = evaluate_itinerary(
        itinerary=current,
        user_state=user_state,
        profile_path=profile_path,
    )
    _log_critic(critic_result, _log)

    if critic_result.get("passed"):
        _log("초기 critic 통과 → 파이프라인 완료")
        log["final_score"] = critic_result.get("overall_score")
        log["final_passed"] = True
        log["cycles"].append({"step": "initial_critic", "result": _summarize_critic(critic_result)})
        return current, log

    log["cycles"].append({"step": "initial_critic", "result": _summarize_critic(critic_result)})

    # ------------------------------------------------
    # Step 2: replanner 사이클
    # ------------------------------------------------
    for replan_cycle in range(MAX_REPLAN_CYCLES):
        needs_replan = critic_result.get("needs_replan", False)
        days_to_replan = critic_result.get("days_needing_replan", [])

        if not needs_replan or not days_to_replan:
            _log("replanner 불필요 — 다음 단계로")
            break

        _log(f"Step 2-{replan_cycle + 1} — replanner 실행 (days={days_to_replan})")
        current, replan_result = replan_itinerary(
            itinerary=current,
            critic_result=critic_result,
            user_state=user_state,
            profile_path=profile_path,
        )

        changed = replan_result.get("changed", False)
        _log(f"replanner {'변경됨' if changed else '변경 없음'}")
        log["cycles"].append({
            "step": f"replanner_{replan_cycle + 1}",
            "result": {
                "changed": changed,
                "actions": len(replan_result.get("actions", [])),
                "warnings": replan_result.get("warnings", []),
            }
        })

        if changed:
            log["total_changes"] += 1

        # replanner 후 재검수
        _log(f"Step 2-{replan_cycle + 1} — critic 재실행")
        critic_result = evaluate_itinerary(
            itinerary=current,
            user_state=user_state,
            profile_path=profile_path,
        )
        _log_critic(critic_result, _log)
        log["cycles"].append({
            "step": f"critic_after_replan_{replan_cycle + 1}",
            "result": _summarize_critic(critic_result),
        })

        if critic_result.get("passed"):
            _log("critic 통과 → repair 단계 스킵")
            break

        if not changed:
            _log("replanner가 변경을 못 함 → 루프 탈출")
            break

    # ------------------------------------------------
    # Step 3: repair 사이클
    # ------------------------------------------------
    for repair_cycle in range(MAX_REPAIR_CYCLES):
        needs_repair = critic_result.get("needs_repair", False)
        days_to_repair = critic_result.get("days_needing_repair", [])

        if not needs_repair or not days_to_repair:
            _log("repair 불필요 — 다음 단계로")
            break

        _log(f"Step 3-{repair_cycle + 1} — repair 실행 (days={days_to_repair})")
        current, repair_result = repair_itinerary(
            itinerary=current,
            critic_result=critic_result,
            profile_path=profile_path,
        )

        changed = repair_result.get("changed", False)
        _log(f"repair {'변경됨' if changed else '변경 없음'}")
        log["cycles"].append({
            "step": f"repair_{repair_cycle + 1}",
            "result": {
                "changed": changed,
                "actions": [
                    {"type": a["action_type"], "status": a["status"], "desc": a["description"]}
                    for a in repair_result.get("actions", [])
                    if a["status"] == "applied"
                ],
            }
        })

        if changed:
            log["total_changes"] += 1

        # repair 후 최종 critic
        _log(f"Step 3-{repair_cycle + 1} — 최종 critic 실행")
        critic_result = evaluate_itinerary(
            itinerary=current,
            user_state=user_state,
            profile_path=profile_path,
        )
        _log_critic(critic_result, _log)
        log["cycles"].append({
            "step": f"critic_after_repair_{repair_cycle + 1}",
            "result": _summarize_critic(critic_result),
        })

        if critic_result.get("passed"):
            _log("최종 critic 통과")
            break

        if not changed:
            _log("repair가 변경을 못 함 → 루프 탈출")
            break

    # ------------------------------------------------
    # 최종 결과
    # ------------------------------------------------
    log["final_score"] = critic_result.get("overall_score")
    log["final_passed"] = critic_result.get("passed", False)
    log["final_issues"] = [
        {
            "day": i.get("day"),
            "type": i.get("issue_type"),
            "severity": i.get("severity"),
            "description": i.get("description"),
        }
        for i in critic_result.get("issues", [])
    ]

    if log["final_passed"]:
        _log(f"파이프라인 완료 — passed=True, score={log['final_score']}")
    else:
        _log(f"파이프라인 완료 — passed=False, score={log['final_score']} (남은 이슈 {len(log['final_issues'])}개)")

    # 파이프라인 로그를 itinerary에 첨부
    current["pipeline_log"] = log

    return current, log


# ============================================================
# 헬퍼
# ============================================================

def _summarize_critic(critic_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall_score": critic_result.get("overall_score"),
        "passed": critic_result.get("passed"),
        "needs_replan": critic_result.get("needs_replan"),
        "needs_repair": critic_result.get("needs_repair"),
        "days_needing_replan": critic_result.get("days_needing_replan", []),
        "days_needing_repair": critic_result.get("days_needing_repair", []),
        "issue_count": len(critic_result.get("issues", [])),
        "issues": [
            {
                "day": i.get("day"),
                "type": i.get("issue_type"),
                "severity": i.get("severity"),
            }
            for i in critic_result.get("issues", [])
        ],
    }


def _log_critic(critic_result: dict[str, Any], log_fn) -> None:
    score = critic_result.get("overall_score")
    passed = critic_result.get("passed")
    issues = critic_result.get("issues", [])
    log_fn(f"  critic → score={score}, passed={passed}, issues={len(issues)}개")
    for issue in issues:
        log_fn(f"    - Day {issue.get('day')} [{issue.get('severity').upper()}] {issue.get('issue_type')}: {issue.get('description', '')[:60]}")


# ============================================================
# CLI
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full critic → replanner → repair pipeline on an itinerary."
    )
    parser.add_argument(
        "--input", type=str, default="",
        help="Input itinerary JSON. Default: output/replanned_itinerary.json",
    )
    parser.add_argument("--output", type=str, default="output/final_itinerary.json")
    parser.add_argument("--profile", type=str, default="output/area_profiles_v2.json")
    parser.add_argument("--purpose", type=str, default="general")
    parser.add_argument("--location", type=str, default="")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = base_dir / input_path
    else:
        input_path = base_dir / "output" / "replanned_itinerary.json"
        if not input_path.exists():
            input_path = base_dir / "output" / "replanned_itinerary_v3_1.json"

    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = base_dir / profile_path

    itinerary = load_json(input_path)
    user_state = {
        "purpose": args.purpose or "general",
        "location": args.location or "",
    }

    print(f"\n[pipeline] 입력: {input_path}")
    print(f"[pipeline] user_state: {user_state}")
    print(f"[pipeline] 실행 시작...\n")

    final_itinerary, log = run_pipeline(
        itinerary=itinerary,
        user_state=user_state,
        profile_path=profile_path,
        verbose=args.verbose,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(final_itinerary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*50}")
    print(f"[pipeline] 최종 결과")
    print(f"  passed       : {log['final_passed']}")
    print(f"  score        : {log['final_score']}")
    print(f"  total_changes: {log['total_changes']}")
    print(f"  output       : {output_path}")

    print(f"\n[pipeline] 실행 단계")
    for cycle in log["cycles"]:
        step = cycle["step"]
        result = cycle["result"]
        if "passed" in result:
            print(f"  [{step}] score={result.get('overall_score')} passed={result.get('passed')} issues={result.get('issue_count', 0)}개")
        elif "changed" in result:
            print(f"  [{step}] changed={result.get('changed')} actions={result.get('actions', 0)}")

    if log.get("final_issues"):
        print(f"\n[pipeline] 남은 이슈")
        for issue in log["final_issues"]:
            print(f"  - Day {issue['day']} [{issue['severity'].upper()}] {issue['type']}: {issue['description'][:60]}")

    print(f"\n[pipeline] 최종 일정")
    for day in final_itinerary.get("days", []):
        print(f"\nDay {day.get('day')} — {day.get('theme')}")
        for p in day.get("pois", []):
            print(
                f"  - {p.get('estimated_start_time', '')}-{p.get('estimated_end_time', '')} "
                f"{p.get('name')} ({p.get('type')})"
            )


if __name__ == "__main__":
    main()
