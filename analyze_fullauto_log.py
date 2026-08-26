"""Turn a full-auto event log into the numbers a field test actually needs.

    python analyze_fullauto_log.py fullauto_events_2026-08-26.jsonl

The headline question is whether the script identifies the section being sung
faster than an operator scanning the monitor. Run PP7_SenseVoice_FullAuto.py with
--shadow during a normally-operated service and this reports, per operator move,
how long the script had already been sure of that slide.

The second question is whether it is *right*. There's no ground truth in a live
service, so this uses the best available proxy: an operator moving the screen
within a few seconds of a script-driven move is the operator undoing it.
"""

import json
import statistics
import sys
from collections import Counter

# An operator move this soon after a script move is almost certainly a correction.
CORRECTION_WINDOW_SEC = 6.0


def load(path):
    events = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️  skipping malformed line {line_no}", file=sys.stderr)
    return events


def pct(part, whole):
    return f"{part / whole:.0%}" if whole else "n/a"


def describe_leads(leads):
    if not leads:
        print("   No operator-driven moves recorded.")
        print("   Run with --shadow during a manually-operated service to collect this.")
        return
    ordered = sorted(leads)
    ahead = [x for x in leads if x > 0]
    print(f"   Operator moves measured : {len(leads)}")
    print(f"   Script was ahead on     : {len(ahead)} ({pct(len(ahead), len(leads))})")
    print(f"   Median lead             : {statistics.median(ordered):+.2f} s")
    print(f"   Mean lead               : {statistics.fmean(ordered):+.2f} s")
    if len(ordered) >= 10:
        print(f"   10th / 90th percentile  : {ordered[len(ordered) // 10]:+.2f} s"
              f" / {ordered[len(ordered) * 9 // 10]:+.2f} s")
    print(f"   Best / worst            : {ordered[-1]:+.2f} s / {ordered[0]:+.2f} s")
    print("   (positive = the script knew the slide before the operator got there)")


def main(path):
    events = load(path)
    if not events:
        print(f"No events in {path}")
        return 1

    kinds = Counter(e['kind'] for e in events)
    locates = [e for e in events if e['kind'] == 'locate']
    moves = [e for e in events if e['kind'] == 'slide_moved']
    actions = [e for e in events if e['kind'] == 'locate_action']
    span = events[-1]['t'] - events[0]['t']

    print(f"\n📄 {path}")
    print(f"   {len(events)} events over {span / 60:.1f} min "
          f"({kinds.get('session_start', 0)} session start(s), "
          f"{kinds.get('song_loaded', 0)} song load(s))")

    print("\n🎯 --- Localization ---")
    confident = sum(1 for e in locates if e.get('confident'))
    print(f"   Decode cycles           : {len(locates)}")
    print(f"   Confidently located     : {confident} ({pct(confident, len(locates))})")
    if locates:
        reasons = Counter(e.get('reason', '') for e in locates if not e.get('confident'))
        for reason, count in reasons.most_common(4):
            label = reason.split(' ')[0] if reason else "(no candidate)"
            print(f"      unsure — {label:<10} {count}")

    print("\n🎬 --- Actions ---")
    taken = Counter(e['action'] for e in actions if e.get('executed'))
    would = Counter(e['action'] for e in actions if not e.get('executed'))
    print(f"   Executed  : {dict(taken) or '—'}")
    print(f"   Suppressed: {dict(would) or '—'}")
    print(f"   Tail advances           : "
          f"{sum(1 for e in events if e['kind'] == 'advance' and e.get('origin') != 'rescue')}")
    print(f"   Advances vetoed as early: {kinds.get('advance_vetoed', 0)}")

    print("\n⏱️  --- Lead Time vs Operator ---")
    describe_leads([e['lead_sec'] for e in moves if e.get('lead_sec') is not None])

    # Accuracy proxy: how often did a human immediately move the screen again
    # after the script moved it?
    print("\n🔍 --- Script moves the operator corrected ---")
    script_moves = [e for e in moves if e.get('source') == 'script']
    corrections = 0
    for i, move in enumerate(moves):
        if move.get('source') != 'script':
            continue
        follow = next((m for m in moves[i + 1:] if m['t'] > move['t']), None)
        if follow and follow.get('source') == 'operator' \
                and follow['t'] - move['t'] <= CORRECTION_WINDOW_SEC:
            corrections += 1
    if script_moves:
        print(f"   {'Script-driven moves':<24}: {len(script_moves)}")
        label = f"Corrected within {CORRECTION_WINDOW_SEC:.0f}s"
        print(f"   {label:<24}: {corrections} ({pct(corrections, len(script_moves))})")
        print("   (a proxy for false positives — lower is better)")
    elif taken:
        # Actions were executed but PP7 never reported the resulting slide change.
        print(f"   {len(list(taken.elements()))} action(s) executed, but no matching")
        print("   slide-change events were logged — can't judge corrections.")
    else:
        print("   None — the script never drove the screen (shadow mode or localizer off).")

    print()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
