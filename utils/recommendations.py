"""Skill-gap analysis and recommendation text for PlaceMentor AI.

The benchmarks below are ILLUSTRATIVE development targets chosen for this
project. They are not official placement eligibility criteria.
"""
from __future__ import annotations

from .clustering import CAREER_READY, DEVELOPING, PRIORITY
from .preprocessing import FEATURES

BENCHMARKS = {
    "CGPA": 7.5,
    "Aptitude": 65,
    "Coding": 65,
    "Communication": 65,
    "Projects": 3,
    "Internships": 1,
    "Attendance": 75,
}

AREAS = {
    "CGPA": "Academic improvement",
    "Aptitude": "Aptitude practice",
    "Coding": "Coding skills",
    "Communication": "Communication skills",
    "Projects": "Project development",
    "Internships": "Internship experience",
    "Attendance": "Attendance",
}

STATUS_LABELS = {
    "priority": "Needs focus",
    "close": "Close to benchmark",
    "strong": "Meets benchmark",
}

GROUP_SUMMARIES = {
    CAREER_READY: (
        "Your profile is closest to students with strong academics, high skill "
        "scores, several projects and internship exposure. Keep building depth "
        "and keep your portfolio current."
    ),
    DEVELOPING: (
        "Your profile is closest to students with a solid base and clear room "
        "to grow. Targeted practice in one or two weaker areas can move you "
        "noticeably forward."
    ),
    PRIORITY: (
        "Your profile is closest to students who are at an early stage in "
        "several areas. A steady weekly routine in coding, aptitude and "
        "communication is the best place to start."
    ),
}


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def _trim(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _value_text(feat: str, v: float) -> str:
    if feat == "CGPA":
        return f"{v:.2f}"
    if feat in ("Projects", "Internships"):
        return str(int(round(v)))
    if feat == "Attendance":
        return _trim(v) + "%"
    return _trim(v)


def _gap_text(feat: str, gap: float) -> str:
    gap = abs(gap)
    if feat == "CGPA":
        return f"{gap:.2f} points"
    if feat == "Projects":
        n = int(round(gap))
        return f"{n} project" + ("" if n == 1 else "s")
    if feat == "Internships":
        n = int(round(gap))
        return f"{n} internship" + ("" if n == 1 else "s")
    if feat == "Attendance":
        return _trim(gap) + " percentage points"
    return _trim(gap) + " points"


# --------------------------------------------------------------------------
# Suggestion templates (filled with the student's actual numbers)
# --------------------------------------------------------------------------
TEMPLATES = {
    "CGPA": {
        "priority": "Your CGPA of {value} is {gap} below the illustrative benchmark of {benchmark}. Find the subjects that pulled it down, use faculty doubt-clearing sessions, and plan fixed weekly revision slots so upcoming semester grades improve.",
        "close": "Your CGPA of {value} is close to the illustrative benchmark of {benchmark}, only {gap} away. Regular revision and on-time assignments can close the remaining difference.",
        "strong": "Your CGPA of {value} meets the illustrative benchmark of {benchmark}. Maintain it while adding practical skills, since marks alone show little about applied ability.",
    },
    "Aptitude": {
        "priority": "Your aptitude score of {value} is {gap} below the illustrative benchmark of {benchmark}. Practise quantitative, logical-reasoning and verbal questions for 30 to 45 minutes daily and take one timed mock test each week to build speed.",
        "close": "Your aptitude score of {value} is {gap} short of the illustrative benchmark of {benchmark}. Identify the question types where you lose time and drill those first.",
        "strong": "Your aptitude score of {value} meets the illustrative benchmark of {benchmark}. Keep it sharp with one timed practice set each week.",
    },
    "Coding": {
        "priority": "Your coding score of {value} is {gap} below the illustrative benchmark of {benchmark}. Solve one or two easy problems daily, revise arrays, strings, recursion and basic data structures, and write one small program from scratch each week.",
        "close": "Your coding score of {value} is {gap} short of the illustrative benchmark of {benchmark}. Move from easy to medium problems and practise explaining your solution aloud.",
        "strong": "Your coding score of {value} meets the illustrative benchmark of {benchmark}. Deepen it with data structures, algorithms and a larger project that uses them.",
    },
    "Communication": {
        "priority": "Your communication score of {value} is {gap} below the illustrative benchmark of {benchmark}. Speak on a topic for two minutes daily, join a seminar or debate club, and record mock introductions to review clarity and confidence.",
        "close": "Your communication score of {value} is {gap} short of the illustrative benchmark of {benchmark}. Practise group discussions and structured answers to common interview questions.",
        "strong": "Your communication score of {value} meets the illustrative benchmark of {benchmark}. Use it by presenting your projects to peers and mentoring juniors.",
    },
    "Projects": {
        "priority": "You have completed {value} project(s), {gap} short of the illustrative benchmark of {benchmark}. Pick a problem you care about, build it end to end, and publish the code with a clear README on GitHub.",
        "close": "You have completed {value} projects, {gap} short of the illustrative benchmark of {benchmark}. Finish one more project and document what you learned.",
        "strong": "You have completed {value} projects, which meets the illustrative benchmark of {benchmark}. Focus on quality: clear documentation, tests and a short demo of each project.",
    },
    "Internships": {
        "priority": "Your internship count of {value} is {gap} below the illustrative benchmark of {benchmark}. Look for virtual internships, open-source contributions, industry-linked college programmes or short summer internships to gain workplace exposure.",
        "close": "Your internship count of {value} is {gap} below the illustrative benchmark of {benchmark}. Seek one more short internship or industry project.",
        "strong": "Your internship count of {value} meets the illustrative benchmark of {benchmark}. Reflect on what you learned and be ready to describe your contribution clearly.",
    },
    "Attendance": {
        "priority": "Your attendance of {value} is {gap} below the illustrative benchmark of {benchmark}. Regular attendance supports continuity of learning, so plan your week to avoid missed classes and speak to faculty about recurring barriers.",
        "close": "Your attendance of {value} is {gap} below the illustrative benchmark of {benchmark}. A few more consistent weeks will bring it up.",
        "strong": "Your attendance of {value} meets the illustrative benchmark of {benchmark}. Keep the routine going.",
    },
}


def _tier(ratio: float) -> str:
    if ratio < 0.85:
        return "priority"
    if ratio < 1.0:
        return "close"
    return "strong"


def analyse_skill_gaps(values: dict) -> dict:
    """Compare a student's entered values with the illustrative benchmarks."""
    items = []
    for feat in FEATURES:
        value = float(values[feat])
        benchmark = BENCHMARKS[feat]
        ratio = value / benchmark if benchmark else 1.0
        tier = _tier(ratio)
        text = TEMPLATES[feat][tier].format(
            value=_value_text(feat, value),
            benchmark=_value_text(feat, benchmark),
            gap=_gap_text(feat, benchmark - value),
        )
        items.append({
            "Feature": feat,
            "Area": AREAS[feat],
            "Your value": value,
            "Benchmark": benchmark,
            "Difference": round(value - benchmark, 2),
            "Percent of benchmark": round(ratio * 100, 1),
            "Tier": tier,
            "Status": STATUS_LABELS[tier],
            "Suggestion": text,
        })

    ordered = sorted(items, key=lambda r: r["Percent of benchmark"])
    focus = [r["Area"] for r in ordered if r["Tier"] != "strong"][:3]
    met = sum(1 for r in items if r["Tier"] == "strong")
    return {
        "items": items,
        "prioritised": ordered,
        "met": met,
        "total": len(items),
        "focus": focus,
    }
