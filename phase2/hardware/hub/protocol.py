"""The recording session, block by block. One list used by everything:
the live runner (prompts + LED commands), the simulator, and the labeller.

Times are seconds from when the node was reset (session start). The LED value is
the MAX30102 IR drive setting (31 = 0x1F = normal). Lowering it is a controlled,
physical "weak sensor" fault: the light really gets dimmer, so the signal really
gets smaller, exactly like a failing LED or a loose clip.
"""
LED_NORMAL = 31

# (start_s, end_s, block, class, level, instruction, led)
BLOCKS = [
    (0, 125, "calibrate", "CLEAN", 0, "Sit still, forearm resting on the table, clip firm. No talking.", LED_NORMAL),
    (125, 200, "clean_1", "CLEAN", 0, "Stay still. This is the clean reference.", LED_NORMAL),
    (200, 230, "noisy_light", "NOISY", 1, "Tap the finger gently on the table, about twice a second.", LED_NORMAL),
    (230, 260, "noisy_strong", "NOISY", 3, "Tap and wiggle the finger strongly.", LED_NORMAL),
    (260, 290, "rest_1", "CLEAN", 0, "Stop. Stay still.", LED_NORMAL),
    (290, 320, "weak_led_1", "ATTENUATED", 1, "Stay still (the LED is being dimmed to 70%).", 22),
    (320, 350, "weak_led_2", "ATTENUATED", 2, "Stay still (LED at 40%).", 12),
    (350, 380, "weak_led_3", "ATTENUATED", 3, "Stay still (LED at 15%).", 5),
    (380, 410, "rest_2", "CLEAN", 0, "Stay still (LED back to normal).", LED_NORMAL),
    (410, 470, "arm_wander", "DRIFTED", 2, "Slowly raise the hand above the heart, then lower it below the table: 10 s each way.", LED_NORMAL),
    (470, 500, "rest_3", "CLEAN", 0, "Hand back on the table. Stay still.", LED_NORMAL),
    (500, 560, "cable_knocks", "SPIKED", 2, "Flick the sensor cable sharply once every 3 seconds.", LED_NORMAL),
    (560, 590, "rest_4", "CLEAN", 0, "Stop. Stay still.", LED_NORMAL),
    (590, 830, "slow_fade", "CLEAN", 0, "Stay still for 4 minutes. The LED fades slowly (drift monitor test).", "ramp"),
    (830, 890, "clean_close", "CLEAN", 0, "Stay still. Last minute.", LED_NORMAL),
]
SESSION_S = BLOCKS[-1][1]
FADE_START, FADE_END, FADE_TO = 590, 830, 22        # 100% -> 70% over 4 minutes
GUARD_S = 3                                          # windows touching a block edge are not labelled


def led_at(t):
    """LED setting the protocol asks for at time t (s)."""
    for start, end, _, _, _, _, led in BLOCKS:
        if start <= t < end:
            if led == "ramp":
                frac = (t - FADE_START) / (FADE_END - FADE_START)
                return int(round(LED_NORMAL + frac * (FADE_TO - LED_NORMAL)))
            return led
    return LED_NORMAL


def block_at(t):
    for b in BLOCKS:
        if b[0] <= t < b[1]:
            return b
    return None
