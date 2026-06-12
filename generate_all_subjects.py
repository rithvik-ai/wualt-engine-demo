"""Generate one synthetic day for each of 5 subjects with different profiles.

Each subject's day is written to subjects/{id}.json (1440 minute-frames).
A subjects.json index file is written with metadata for the picker.

Profiles
--------
S01  Priya Singh,   28 F, office worker      — assault scenario at 15:05
S02  Rakesh Patel,  52 M, truck driver       — highway scare at 04:30
S03  Mrs Iyer,      71 F, retired, alone     — kitchen fall + medical at 19:15
S04  Vikram Mehta,  35 M, anxiety patient    — panic attack at 09:35
S05  Anika Sharma,  24 F, night-shift nurse  — workplace emergency at 05:30
"""
from __future__ import annotations
import os, json, math, random


def jitter(rng, v, sigma):
    return round(v + rng.gauss(0, sigma), 2)


def gen_priya(rng, minute, baseline) -> dict:
    """28F office worker — afternoon assault."""
    hr_h = minute // 60
    is_night = (hr_h < 7) or (hr_h >= 22)
    if minute < 360:
        hr, hrv, spo2, temp = 60, 85, 97.3, 36.35; motion, acc = "still", 0.005
        audio, danger, geo, expected, label = "silence", 0.02, "home", "normal", "Deep sleep"
    elif minute < 420:
        t = (minute-360)/60
        hr, hrv, spo2, temp = 60+12*t, 85-30*t, 98, 36.4+0.2*t; motion, acc = ("still" if t<0.5 else "walking"), 0.01+0.03*t
        audio, danger, geo, expected, label = "silence", 0.05, "home", "normal", "Waking up"
    elif minute < 450:
        t = (minute-420)/30
        hr, hrv, spo2, temp = 130+12*math.sin(t*math.pi), 28, 97.5, 37.0; motion, acc = "exercise", 0.55
        audio, danger, geo, expected, label = ("traffic" if t>0.3 else "silence"), 0.10, "known", "normal", "Morning jog (HR elevated, exercise suppresses)"
    elif minute < 540:
        t = (minute-450)/90
        hr, hrv, spo2, temp = 95-23*t, 40+15*t, 98, 36.9-0.3*t; motion, acc = ("walking" if t<0.5 else "still"), 0.08-0.04*t
        audio, danger, geo, expected, label = "speech", 0.10, "home", "normal", "Cool-down / breakfast"
    elif minute < 720:
        spike = (minute-540) % 47 == 0
        hr, hrv = 78+(12 if spike else 0), 50-(10 if spike else 0); spo2, temp = 98, 36.6; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.15, "work", "normal", "Morning at work" + (" (brief stress)" if spike else "")
    elif minute < 780:
        hr, hrv, spo2, temp = 80, 50, 98, 36.7; motion, acc = ("walking" if rng.random()<0.5 else "still"), 0.06
        audio, danger, geo, expected, label = "speech", 0.12, "known", "normal", "Lunch break"
    elif minute < 840:
        hr, hrv, spo2, temp = 76, 52, 98, 36.6; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.12, "work", "normal", "Afternoon work"
    elif minute < 900:
        hr, hrv, spo2, temp = 83, 53, 98, 36.7; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.15, "work", "stress", "Important meeting (mild stress)"
    elif minute < 903:
        hr, hrv, spo2, temp = 82, 45, 98, 36.7; motion, acc = "walking", 0.12
        audio, danger, geo, expected, label = "traffic", 0.18, "known", "normal", "Walking to car"
    elif minute == 903:
        hr, hrv, spo2, temp = 118, 28, 98, 36.7; motion, acc = "fall", 4.8
        audio, danger, geo, expected, label = "thud", 0.35, "known", "stress", "★ FALL — slip at kerb"
    elif minute < 905:
        hr, hrv, spo2, temp = 102, 34, 98, 36.8; motion, acc = "walking", 0.18
        audio, danger, geo, expected, label = "traffic", 0.22, "known", "stress", "Getting up, shaken but mobile"
    elif minute < 935:
        progress = (minute - 905) / 30
        if progress < 0.27:
            t = progress / 0.27
            hr, hrv, spo2, temp = 95+40*t, 45-33*t, 98-3*t, 36.7+0.6*t
            audio = "scream" if progress < 0.15 else "speech"
            danger = 0.78 if progress < 0.15 else 0.40
            motion, acc = ("still" if progress > 0.10 else "walking"), (0.05 if progress > 0.10 else 0.15)
            label = "★ DISTRESS — approached in parking lot, scream detected"
        else:
            t = (progress - 0.27) / 0.73
            hr, hrv, spo2, temp = 135-50*t, 12+30*t, 95+3*t, 37.3-0.5*t
            audio, danger = "speech", 0.35 - 0.20*t
            motion, acc = "still", 0.05
            label = "★ DISTRESS — recovering, security arrived"
        geo, expected = "unfamiliar", "distress"
    elif minute < 990:
        t = (minute-935)/55
        hr, hrv, spo2, temp = 85-10*t, 38+10*t, 98, 36.8-0.1*t; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.20, "known", "stress", "Recovering, statement to security"
    elif minute < 1080:
        hr, hrv, spo2, temp = 76, 48, 98, 36.6; motion, acc = ("vehicle" if rng.random()<0.6 else "walking"), 0.06
        audio, danger, geo, expected, label = "traffic", 0.15, "known", "normal", "Heading home"
    elif minute < 1200:
        hr, hrv, spo2, temp = 74, 54, 98, 36.6; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.10, "home", "normal", "Dinner with family"
    elif minute < 1320:
        hr, hrv, spo2, temp = 70, 58, 98, 36.5; motion, acc = "still", 0.01
        audio, danger, geo, expected, label = "music", 0.05, "home", "normal", "Watching TV"
    elif minute < 1410:
        t = (minute-1320)/90
        hr, hrv, spo2, temp = 68-5*t, 62+15*t, 98, 36.5-0.05*t; motion, acc = "still", 0.005
        audio, danger, geo, expected, label = "silence", 0.02, "home", "normal", "Reading"
    else:
        t = (minute-1410)/30
        hr, hrv, spo2, temp = 63-3*t, 77+8*t, 97.5, 36.4; motion, acc = "still", 0.005
        audio, danger, geo, expected, label = "silence", 0.02, "home", "normal", "Sleep onset"
    return _build(rng, minute, hr_h, is_night, hr, hrv, spo2, temp, motion, acc, audio, danger, geo, expected, label)


def gen_rakesh(rng, minute, baseline) -> dict:
    """52F domestic worker, Bengaluru — pre-dawn followed-by-stranger scenario at 04:33."""
    hr_h = minute // 60
    is_night = (hr_h < 7) or (hr_h >= 22)
    if minute < 180:
        hr, hrv, spo2, temp = 64, 50, 97.5, 36.4; motion, acc = "still", 0.005
        audio, danger, geo, expected, label = "silence", 0.02, "home", "normal", "Sleep at home (Yelahanka chawl)"
    elif minute < 240:
        t = (minute-180)/60
        hr, hrv, spo2, temp = 64+14*t, 50-12*t, 98, 36.5; motion, acc = "still", 0.01
        audio, danger, geo, expected, label = "silence", 0.03, "home", "normal", "Pre-dawn wake-up · tea + tiffin"
    elif minute < 265:
        # 04:00 — walking to bus stop
        hr, hrv, spo2, temp = 84, 42, 98, 36.6; motion, acc = "walking", 0.14
        audio, danger, geo, expected, label = "traffic", 0.20, "known", "normal", "Walking to early-morning bus stop"
    elif minute < 270:
        # 04:25 — bus ride
        hr, hrv, spo2, temp = 78, 44, 98, 36.6; motion, acc = "vehicle", 0.08
        audio, danger, geo, expected, label = "speech", 0.18, "known", "normal", "BMTC bus · headed to Jayanagar"
    elif minute < 273:
        # 04:30 — got off bus, walking the last stretch alone
        hr, hrv, spo2, temp = 88, 32, 98, 36.7; motion, acc = "walking", 0.16
        audio, danger, geo, expected, label = "footsteps", 0.35, "unfamiliar", "stress", "Walking alone · noticed someone following"
    elif minute < 275:
        # 04:33 — DISTRESS: brief acute fear
        hr, hrv, spo2, temp = 132, 14, 96, 37.2; motion, acc = "walking", 0.20
        audio, danger, geo, expected, label = "shout", 0.65, "unfamiliar", "distress", "★ DISTRESS — followed by stranger · shouting for help"
    elif minute < 285:
        t = (minute-275)/10
        hr, hrv, spo2, temp = 132-32*t, 14+24*t, 96+1.5*t, 37.2-0.4*t; motion, acc = "walking", 0.10
        audio, danger, geo, expected, label = "speech", 0.30, "unfamiliar", "stress", "Adrenaline crash · auto driver helped"
    elif minute < 420:
        hr, hrv, spo2, temp = 82, 40, 98, 36.7; motion, acc = "walking", 0.08
        audio, danger, geo, expected, label = "speech", 0.18, "known", "normal", "First house · cleaning"
    elif minute < 720:
        hr, hrv, spo2, temp = 80, 42, 98, 36.7; motion, acc = "walking", 0.10
        audio, danger, geo, expected, label = "speech", 0.15, "known", "normal", "Second + third house · morning rounds"
    elif minute < 780:
        hr, hrv, spo2, temp = 76, 48, 98, 36.6; motion, acc = ("walking" if rng.random()<0.5 else "still"), 0.05
        audio, danger, geo, expected, label = "speech", 0.10, "known", "normal", "Lunch break · idli at corner stall"
    elif minute < 1020:
        hr, hrv, spo2, temp = 82, 42, 98, 36.7; motion, acc = "walking", 0.10
        audio, danger, geo, expected, label = "speech", 0.15, "known", "normal", "Afternoon rounds · fourth + fifth house"
    elif minute < 1080:
        # 17:00–18:00 STRESS — confrontation with employer about delayed wages
        hr, hrv, spo2, temp = 84, 53, 98, 36.7; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.22, "known", "stress", "Employer confrontation · delayed wages"
    elif minute < 1260:
        hr, hrv, spo2, temp = 74, 46, 98, 36.6; motion, acc = ("vehicle" if rng.random()<0.5 else "walking"), 0.06
        audio, danger, geo, expected, label = "speech", 0.10, "known", "normal", "Bus back · dinner with family"
    elif minute < 1440:
        hr, hrv, spo2, temp = 64, 50, 97.5, 36.4; motion, acc = "still", 0.005
        audio, danger, geo, expected, label = "silence", 0.02, "home", "normal", "Sleep at home"
    else:
        hr, hrv, spo2, temp = 64, 50, 97.5, 36.4; motion, acc = "still", 0.005
        audio, danger, geo, expected, label = "silence", 0.02, "home", "normal", "Sleep"
    return _build(rng, minute, hr_h, is_night, hr, hrv, spo2, temp, motion, acc, audio, danger, geo, expected, label)


def gen_iyer(rng, minute, baseline) -> dict:
    """71F retired teacher, lives alone — kitchen fall at 19:15."""
    hr_h = minute // 60
    is_night = (hr_h < 7) or (hr_h >= 22)
    if minute < 360:
        hr, hrv, spo2, temp = 62, 42, 96.5, 36.3; motion, acc = "still", 0.005
        audio, danger, geo, expected, label = "silence", 0.02, "home", "normal", "Sleep"
    elif minute < 420:
        t = (minute-360)/60
        hr, hrv, spo2, temp = 62+10*t, 42-2*t, 97, 36.4+0.1*t; motion, acc = ("still" if t<0.5 else "walking"), 0.01+0.02*t
        audio, danger, geo, expected, label = "silence", 0.05, "home", "normal", "Slow waking"
    elif minute < 480:
        hr, hrv, spo2, temp = 76, 38, 97, 36.5; motion, acc = "walking", 0.04
        audio, danger, geo, expected, label = "silence", 0.05, "home", "normal", "Morning tea + medication"
    elif minute < 540:
        hr, hrv, spo2, temp = 92, 32, 97.5, 36.7; motion, acc = "walking", 0.15
        audio, danger, geo, expected, label = "traffic", 0.10, "known", "normal", "Morning walk in colony"
    elif minute < 600:
        hr, hrv, spo2, temp = 68, 42, 97, 36.5; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "music", 0.05, "home", "normal", "Reading / newspaper"
    elif minute < 660:
        # 10:00–11:00 STRESS — looking at late husband's photos, anniversary
        hr, hrv, spo2, temp = 80, 53, 97, 36.5; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "music", 0.08, "home", "stress", "Looking at husband's photos · wedding anniversary"
    elif minute < 720:
        hr, hrv, spo2, temp = 68, 42, 97, 36.5; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "music", 0.05, "home", "normal", "Reading / newspaper"
    elif minute < 780:
        hr, hrv, spo2, temp = 72, 40, 97, 36.5; motion, acc = ("walking" if rng.random()<0.5 else "still"), 0.05
        audio, danger, geo, expected, label = "speech", 0.10, "home", "normal", "Lunch · prepared at home"
    elif minute < 960:
        hr, hrv, spo2, temp = 66, 44, 97, 36.4; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "music", 0.05, "home", "normal", "Afternoon TV / rest"
    elif minute < 1155:
        hr, hrv, spo2, temp = 70, 42, 97, 36.4; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "music", 0.05, "home", "normal", "Reading / TV"
    elif minute < 1160:
        hr, hrv, spo2, temp = 80, 38, 97, 36.5; motion, acc = "walking", 0.10
        audio, danger, geo, expected, label = "speech", 0.10, "home", "normal", "Cooking dinner"
    elif minute == 1160:
        # 19:20 — kitchen fall
        hr, hrv, spo2, temp = 110, 25, 96, 36.6; motion, acc = "fall", 5.3
        audio, danger, geo, expected, label = "thud", 0.42, "home", "stress", "★ FALL — kitchen slip, hard impact"
    elif minute < 1165:
        t = (minute-1160)/5
        hr, hrv, spo2, temp = 110+12*t, 25-12*t, 96-2*t, 36.6+0.2*t; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.30, "home", "distress", "★ DISTRESS — on floor, can't get up"
    elif minute < 1175:
        t = (minute-1165)/10
        hr, hrv, spo2, temp = 122-8*t, 13+8*t, 94+1.5*t, 36.8-0.05*t; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.25, "home", "distress", "★ DISTRESS — calling for help"
    elif minute < 1200:
        t = (minute-1175)/25
        hr, hrv, spo2, temp = 114-22*t, 21+15*t, 95.5+1.5*t, 36.75-0.1*t; motion, acc = ("still" if t<0.5 else "walking"), 0.05
        audio, danger, geo, expected, label = "speech", 0.20, "home", "stress", "Recovering, neighbour arrived"
    elif minute < 1320:
        hr, hrv, spo2, temp = 70, 40, 97, 36.4; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.10, "home", "normal", "Neighbour stayed · light dinner"
    elif minute < 1440:
        hr, hrv, spo2, temp = 64, 42, 97, 36.3; motion, acc = "still", 0.005
        audio, danger, geo, expected, label = "silence", 0.02, "home", "normal", "Sleep"
    else:
        hr, hrv, spo2, temp = 64, 42, 97, 36.3; motion, acc = "still", 0.005
        audio, danger, geo, expected, label = "silence", 0.02, "home", "normal", "Sleep"
    return _build(rng, minute, hr_h, is_night, hr, hrv, spo2, temp, motion, acc, audio, danger, geo, expected, label)


def gen_vikram(rng, minute, baseline) -> dict:
    """35M sales executive, anxiety disorder — panic attack at 09:35 during pitch."""
    hr_h = minute // 60
    is_night = (hr_h < 7) or (hr_h >= 22)
    if minute < 360:
        hr, hrv, spo2, temp = 65, 60, 98, 36.4; motion, acc = "still", 0.005
        audio, danger, geo, expected, label = "silence", 0.02, "home", "normal", "Sleep"
    elif minute < 420:
        t = (minute-360)/60
        hr, hrv, spo2, temp = 65+11*t, 60-18*t, 98, 36.5+0.1*t; motion, acc = ("still" if t<0.5 else "walking"), 0.01+0.03*t
        audio, danger, geo, expected, label = "silence", 0.05, "home", "normal", "Waking up"
    elif minute < 480:
        hr, hrv, spo2, temp = 80, 38, 98, 36.7; motion, acc = "walking", 0.10
        audio, danger, geo, expected, label = "speech", 0.15, "home", "normal", "Getting ready, anticipatory stress"
    elif minute < 540:
        hr, hrv, spo2, temp = 84, 35, 98, 36.7; motion, acc = "vehicle", 0.08
        audio, danger, geo, expected, label = "traffic", 0.20, "known", "stress", "Commute to office · pitch today"
    elif minute < 570:
        # 09:00 — at office, waiting for pitch
        hr, hrv, spo2, temp = 92, 28, 98, 36.8; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.20, "work", "stress", "Pre-pitch waiting room"
    elif minute < 575:
        # 09:30 — pitch starts
        hr, hrv, spo2, temp = 105, 22, 97.5, 36.9; motion, acc = "still", 0.03
        audio, danger, geo, expected, label = "speech", 0.30, "work", "stress", "Pitch starting"
    elif minute < 583:
        # 09:35 — PANIC ATTACK
        t = (minute-575)/8
        hr, hrv, spo2, temp = 105+30*t, 22-10*t, 97.5-1.5*t, 36.9+0.4*t; motion, acc = "still", 0.04
        audio, danger, geo, expected, label = "speech", 0.40, "work", "distress", "★ DISTRESS — panic attack mid-pitch"
    elif minute < 600:
        t = (minute-583)/17
        hr, hrv, spo2, temp = 135-40*t, 12+22*t, 96+2*t, 37.3-0.3*t; motion, acc = ("still" if t<0.4 else "walking"), 0.05
        audio, danger, geo, expected, label = "speech", 0.25, "work", "stress", "Stepped out · recovering"
    elif minute < 720:
        hr, hrv, spo2, temp = 78, 45, 98, 36.6; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.15, "work", "normal", "Quiet office work · resting"
    elif minute < 780:
        hr, hrv, spo2, temp = 76, 50, 98, 36.6; motion, acc = ("walking" if rng.random()<0.5 else "still"), 0.05
        audio, danger, geo, expected, label = "speech", 0.10, "known", "normal", "Lunch"
    elif minute < 1020:
        hr, hrv, spo2, temp = 75, 52, 98, 36.6; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.12, "work", "normal", "Afternoon work"
    elif minute < 1080:
        # 17:00–18:00 STRESS — reviewing morning recording, anxiety about tomorrow's follow-up
        hr, hrv, spo2, temp = 83, 53, 98, 36.7; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.20, "work", "stress", "Reviewing pitch · anxious about tomorrow"
    elif minute < 1200:
        hr, hrv, spo2, temp = 72, 55, 98, 36.5; motion, acc = ("vehicle" if rng.random()<0.5 else "walking"), 0.07
        audio, danger, geo, expected, label = "traffic", 0.10, "known", "normal", "Commute home"
    elif minute < 1320:
        hr, hrv, spo2, temp = 70, 58, 98, 36.5; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.08, "home", "normal", "Evening at home"
    elif minute < 1440:
        hr, hrv, spo2, temp = 65, 60, 98, 36.4; motion, acc = "still", 0.005
        audio, danger, geo, expected, label = "silence", 0.02, "home", "normal", "Sleep onset"
    else:
        hr, hrv, spo2, temp = 65, 60, 98, 36.4; motion, acc = "still", 0.005
        audio, danger, geo, expected, label = "silence", 0.02, "home", "normal", "Sleep"
    return _build(rng, minute, hr_h, is_night, hr, hrv, spo2, temp, motion, acc, audio, danger, geo, expected, label)


def gen_sushma(rng, minute, baseline) -> dict:
    """45F yoga instructor — peaceful day, NO distress event.
       Demonstrates the engine's specificity: doesn't false-alarm during exercise
       or daily-life mild stress. Used to show 'engine stays quiet when it should'.
    """
    hr_h = minute // 60
    is_night = (hr_h < 7) or (hr_h >= 22)

    # 00:00–06:00 sleep
    if minute < 360:
        hr, hrv, spo2, temp = 58, 78, 97.5, 36.3; motion, acc = "still", 0.005
        audio, danger, geo, expected, label = "silence", 0.02, "home", "normal", "Sleep"
    # 06:00–06:45 waking up
    elif minute < 405:
        t = (minute-360)/45
        hr, hrv, spo2, temp = 58+10*t, 78-15*t, 98, 36.4+0.2*t; motion, acc = ("still" if t<0.5 else "walking"), 0.01+0.03*t
        audio, danger, geo, expected, label = "silence", 0.05, "home", "normal", "Waking up · tea"
    # 06:45–08:00 morning meditation + setup
    elif minute < 480:
        hr, hrv, spo2, temp = 64, 70, 98, 36.5; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "music", 0.05, "home", "normal", "Personal meditation"
    # 08:00–09:30 morning yoga class (she teaches) — EXERCISE
    #   HR elevated because of demonstration poses, but motion=exercise
    #   so the engine suppresses HR-only flag → state stays normal
    elif minute < 570:
        t = (minute-480)/90
        hr, hrv, spo2, temp = 110 + 8*math.sin(t*math.pi*2), 35, 97.5, 37.0; motion, acc = "exercise", 0.55
        audio, danger, geo, expected, label = "music", 0.10, "home", "normal", "Teaching morning yoga · demonstrations"
    # 09:30–10:00 cool-down + tea with students
    elif minute < 600:
        t = (minute-570)/30
        hr, hrv, spo2, temp = 95-25*t, 50+20*t, 98, 36.8-0.2*t; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.10, "home", "normal", "Cool-down · tea with students"
    # 10:00–12:00 admin + emails (brief stress spikes at deadlines)
    elif minute < 720:
        spike = (minute-600) % 53 == 0
        hr, hrv = 72+(8 if spike else 0), 60-(8 if spike else 0); spo2, temp = 98, 36.5; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.10, "home", "normal", "Studio admin" + (" (brief deadline)" if spike else "")
    # 12:00–13:00 lunch (calm, with family)
    elif minute < 780:
        hr, hrv, spo2, temp = 70, 65, 98, 36.6; motion, acc = ("walking" if rng.random()<0.3 else "still"), 0.04
        audio, danger, geo, expected, label = "speech", 0.08, "home", "normal", "Lunch · family"
    # 13:00–15:00 short nap + reading
    elif minute < 900:
        hr, hrv, spo2, temp = 62, 75, 98, 36.4; motion, acc = "still", 0.005
        audio, danger, geo, expected, label = "silence", 0.02, "home", "normal", "Short nap / reading"
    # 15:00–16:00 admin call with studio team — mild work stress (15:30 stress block)
    elif minute < 960:
        in_call = (minute >= 930)            # 15:30 — slightly tense team call
        hr, hrv = (83 if in_call else 75), (53 if in_call else 60); spo2, temp = 98, 36.6; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.15, "home", ("stress" if in_call else "normal"), \
            ("Studio team call · disagreement" if in_call else "Reviewing class plans")
    # 16:00–17:00 walk to evening session (light cardio)
    elif minute < 1020:
        hr, hrv, spo2, temp = 92, 42, 98, 36.7; motion, acc = "walking", 0.15
        audio, danger, geo, expected, label = "traffic", 0.10, "known", "normal", "Walk to evening session"
    # 17:00–18:30 evening private yoga session — EXERCISE again
    elif minute < 1110:
        hr, hrv, spo2, temp = 105 + 6*math.sin((minute-1020)/90*math.pi*2), 38, 97.5, 37.0; motion, acc = "exercise", 0.50
        audio, danger, geo, expected, label = "music", 0.08, "known", "normal", "Evening private session"
    # 18:30–19:30 walk home + cool down
    elif minute < 1170:
        t = (minute-1110)/60
        hr, hrv, spo2, temp = 92-22*t, 48+15*t, 98, 36.8-0.2*t; motion, acc = ("walking" if t<0.7 else "still"), 0.08
        audio, danger, geo, expected, label = "traffic", 0.10, "home", "normal", "Walk home · cool down"
    # 19:30–21:00 dinner + family time
    elif minute < 1260:
        hr, hrv, spo2, temp = 72, 60, 98, 36.6; motion, acc = "still", 0.02
        audio, danger, geo, expected, label = "speech", 0.08, "home", "normal", "Dinner with family"
    # 21:00–22:30 reading + winding down
    elif minute < 1350:
        hr, hrv, spo2, temp = 66, 70, 98, 36.5; motion, acc = "still", 0.005
        audio, danger, geo, expected, label = "silence", 0.02, "home", "normal", "Reading · winding down"
    # 22:30–24:00 sleep onset
    else:
        t = min(1.0, (minute-1350)/90)
        hr, hrv, spo2, temp = 64-6*t, 75+5*t, 97.5, 36.4-0.05*t; motion, acc = "still", 0.005
        audio, danger, geo, expected, label = "silence", 0.02, "home", "normal", "Sleep onset"
    return _build(rng, minute, hr_h, is_night, hr, hrv, spo2, temp, motion, acc, audio, danger, geo, expected, label)


def _build(rng, minute, hr_h, is_night, hr, hrv, spo2, temp, motion, acc, audio, danger, geo, expected, label):
    hr_m = minute % 60
    hr   = jitter(rng, hr,   1.5)
    hrv  = jitter(rng, hrv,  2.0)
    spo2 = jitter(rng, spo2, 0.25)
    temp = jitter(rng, temp, 0.05)
    acc_mag_val = 5.2 if motion == "fall" else round(0.98 + rng.gauss(0, 0.005), 3)
    return {
        "minute": minute, "time": f"{hr_h:02d}:{hr_m:02d}", "is_night": is_night,
        "expected_state": expected, "label": label,
        "hr":   max(40, min(180, hr)), "hrv":  max(0, min(120, hrv)),
        "spo2": max(85, min(100, spo2)), "temp": max(33, min(40, temp)),
        "motion": motion, "acc_mag": acc_mag_val, "dyn_acc_mag": round(acc, 3),
        "audio_class": audio, "audio_danger": round(danger, 2), "geo_zone": geo,
        "fall_event": motion == "fall",
    }


SUBJECTS = [
    {"id": "S01", "name": "Priya Singh",       "age": 28, "sex": "F", "context": "Office worker, Bengaluru",
     "baseline_hr": 72, "baseline_hrv": 55, "baseline_spo2": 98, "baseline_temp": 36.6,
     "distress_window": "15:05–15:35", "distress_label": "Approached in parking lot · scream detected",
     "generator": gen_priya, "seed": 42},
    {"id": "S02", "name": "Lakshmi Reddy",     "age": 52, "sex": "F", "context": "Domestic worker · pre-dawn commute, Bengaluru",
     "baseline_hr": 78, "baseline_hrv": 42, "baseline_spo2": 98, "baseline_temp": 36.6,
     "distress_window": "04:33–04:45", "distress_label": "Followed by stranger on dawn commute · shouted for help",
     "generator": gen_rakesh, "seed": 100},
    {"id": "S03", "name": "Mrs Iyer",          "age": 71, "sex": "F", "context": "Retired teacher, lives alone, Mumbai",
     "baseline_hr": 68, "baseline_hrv": 45, "baseline_spo2": 97, "baseline_temp": 36.4,
     "distress_window": "19:20–19:35", "distress_label": "Kitchen fall · on floor · neighbour arrives",
     "generator": gen_iyer, "seed": 200},
    {"id": "S04", "name": "Vidya Iyengar",     "age": 35, "sex": "F", "context": "Sales exec, anxiety disorder, Hyderabad",
     "baseline_hr": 75, "baseline_hrv": 52, "baseline_spo2": 98, "baseline_temp": 36.5,
     "distress_window": "09:35–09:43", "distress_label": "Panic attack mid-pitch",
     "generator": gen_vikram, "seed": 300},
    {"id": "S05", "name": "Sushma Rao",        "age": 45, "sex": "F", "context": "Yoga instructor, home studio · Pune",
     "baseline_hr": 64, "baseline_hrv": 70, "baseline_spo2": 98, "baseline_temp": 36.4,
     "distress_window": "— (no distress event)", "distress_label": "Calm day — meditation, classes, family time. Engine stays silent.",
     "generator": gen_sushma, "seed": 400},
]


def main():
    os.makedirs("subjects", exist_ok=True)
    index = []
    for cfg in SUBJECTS:
        rng = random.Random(cfg["seed"])
        baseline = {"hr": cfg["baseline_hr"], "hrv": cfg["baseline_hrv"],
                    "spo2": cfg["baseline_spo2"], "temp": cfg["baseline_temp"]}
        frames = [cfg["generator"](rng, m, baseline) for m in range(1440)]
        from collections import Counter
        counts = Counter(f["expected_state"] for f in frames)
        out = {
            "profile": {"id": cfg["id"], "name": cfg["name"], "age": cfg["age"], "sex": cfg["sex"],
                        "context": cfg["context"],
                        "baseline_hr": cfg["baseline_hr"], "baseline_hrv": cfg["baseline_hrv"],
                        "baseline_spo2": cfg["baseline_spo2"], "baseline_temp": cfg["baseline_temp"]},
            "frames": frames, "n_frames": len(frames), "summary": dict(counts),
            "distress_window": cfg["distress_window"], "distress_label": cfg["distress_label"],
        }
        path = f"subjects/{cfg['id']}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"  {cfg['id']:5s} {cfg['name']:18s} {cfg['age']}{cfg['sex']:1s}  "
              f"distress @ {cfg['distress_window']:14s}  "
              f"states {dict(counts)}")
        index.append({"id": cfg["id"], "name": cfg["name"], "age": cfg["age"], "sex": cfg["sex"],
                      "context": cfg["context"],
                      "distress_window": cfg["distress_window"], "distress_label": cfg["distress_label"]})
    with open("subjects/_index.json", "w") as f:
        json.dump({"subjects": index}, f, indent=2)
    print(f"\nIndex written: subjects/_index.json ({len(index)} subjects)")


if __name__ == "__main__":
    main()
