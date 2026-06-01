import pyjoycon


class JoyCon(pyjoycon.GyroTrackingJoyCon, pyjoycon.ButtonEventJoyCon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.button_state = set()


joycon = JoyCon(*pyjoycon.get_R_id())


while True:
    print(joycon.pointer, joycon.rotation, joycon.direction, joycon.accel, joycon.gyro)
