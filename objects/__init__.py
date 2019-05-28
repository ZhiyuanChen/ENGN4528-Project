from enum import Enum, unique


class Image(object):
    def __init__(self, image):
        self.image = image
        self.windshield = image[0:490, 0:1200]
        self.left_mirror = image[58:296, 16:193]
        self.right_mirror = image[58:296, 1082:1259]
        self.navigation = image[578:683, 246:424]
        self.assistant = image[268:472, 27:287]
        self.dashboard = image[570:700, 622:1182]


class Dashboard(object):
    def __init__(self, image):
        self.odometer = image[10:60, 87:98]
        self.fuel_gauge = image[612:620, 920:974]
        self.left_turn = image[3:17, 116:132]
        self.right_turn = image[3:17, 244:260]
        self.parking_break = image[2:17, 282:301]
        self.seat_belt = image[2:18, 309:317]
        self.battery_charge = image[10:15, 349:362]
        self.malfunction_indicator = image[115:128, 287:307]
        self.glow_plug = image[116:128, 313:329]
        self.light0 = image[9:15, 83:98]
        self.light1 = image[2:10, 487:498]
        self.light2 = image[4:12, 502:514]
        self.light3 = image[5:12, 517:528]
        self.light4 = image[6:12, 533:548]
        self.light5 = image[7:12, 553:557]
        self.differential0 = image[119:128, 3:20]
        self.differential1 = image[118:128, 26:44]
        self.tandem_axle_lift0 = image[117:128, 49:67]
        self.tandem_axle_lift1 = image[117:127, 72:91]

    @staticmethod
    def is_blue(part):
        indicate = False
        if part[:, :, 0].mean() > 40:
            part[:, :, 0] = 255
            indicate = True
        return indicate

    @staticmethod
    def is_red(part):
        indicate = False
        if part[:, :, 2].mean() > 40:
            part[:, :, 2] = 255
            indicate = True
        return indicate

    @staticmethod
    def is_green(part):
        indicate = False
        if part[:, :, 1].mean() > 40:
            part[:, :, 1] = 255
            indicate = True
        return indicate

    @staticmethod
    def is_yellow(part):
        indicate = False
        if (part[:, :, 1] + part[:, :, 2]).mean() > 80:
            part[:, :, 1] = 255
            part[:, :, 2] = 255
            indicate = True
        return indicate


@unique
class Period(Enum):
    NotSet = 0
    Spring = 1
    Morning = 2
    Noon = 3
    Afternoon = 4
    Fall = 5
    Evening = 6


@unique
class Position(Enum):
    NotSet = 0
    HighSpeed = 1
    State = 2
    City = 3
    Rural = 4
    Tunnel = 5
    Bridge = 6
    Factory = 7
    Harbour = 8
    Train = 9
    Toll = 10
    Fuel = 11


class Truck(object):
    # Engine status: 0 Not Started, 1 Electric Started, 2 Engine Started
    # Turn signal: 0 Off, 1 Left, 2 Right, 3 Emergency
    # Parking brake
    # brake
    # light
    # odometer
    # Fuel gauge
    def __init__(self, seat_belt=0, engine_status=0, turn_signal=0, parking_brake=0, brake=0, differential=0,
                 tandem_axle_lift=0, light=0, odometer=0, fuel_gauge=0, speed_limit=0, navigation=0,
                 position=Position.NotSet, period=Period.NotSet):
        self._seat_belt = seat_belt
        self._engine_status = engine_status
        self._turn_signal = turn_signal
        self._parking_brake = parking_brake
        self._brake = brake
        self._differential = differential
        self._tandem_axle_lift = tandem_axle_lift
        self._light = light
        self._odometer = odometer
        self._fuel_gauge = fuel_gauge
        self._speed_limit = speed_limit
        self._navigation = navigation
        self._position = position
        self._period = period

    def dashboard(self, image):
        dashboard = Dashboard(image)
        if dashboard.is_green(dashboard.left_turn):
            if dashboard.is_green(dashboard.right_turn):
                self.turn_signal = 3
            else:
                self.turn_signal = 1
        elif dashboard.is_green(dashboard.right_turn):
            self.turn_signal = 2
        else:
            self.turn_signal = 0

        if dashboard.is_yellow(dashboard.malfunction_indicator):
            if dashboard.is_yellow(dashboard.battery_charge):
                if dashboard.is_yellow(dashboard.glow_plug):
                    self.engine_status = 1
                else:
                    self.engine_status = 2
            elif dashboard.is_yellow(dashboard.glow_plug):
                self.engine_status = 3
            else:
                self.engine_status = 5
        elif dashboard.is_yellow(dashboard.battery_charge):
            if dashboard.is_yellow(dashboard.glow_plug):
                self.engine_status = 4
            else:
                self.engine_status = 6
        elif dashboard.is_yellow(dashboard.glow_plug):
            self.engine_status = 7
        else:
            self.engine_status = 0

        if dashboard.is_yellow(dashboard.differential0):
            if dashboard.is_yellow(dashboard.differential1):
                self.differential = 1
            else:
                self.differential = 2
        elif dashboard.is_yellow(dashboard.differential1):
            self.differential = 3
        else:
            self.differential = 0

        if dashboard.is_yellow(dashboard.tandem_axle_lift0):
            if dashboard.is_yellow(dashboard.tandem_axle_lift1):
                self.tandem_axle_lift = 1
            else:
                self.tandem_axle_lift = 2
        elif dashboard.is_yellow(dashboard.tandem_axle_lift1):
            self.tandem_axle_lift = 3
        else:
            self.tandem_axle_lift = 0

        self.parking_brake = 1 if dashboard.is_red(dashboard.parking_break) else 0
        self.seat_belt = 1 if dashboard.is_red(dashboard.parking_break) else 0
        return image

    @property
    def seat_belt(self):
        return self._seat_belt

    @seat_belt.setter
    def seat_belt(self, value):
        self._seat_belt = value

    @seat_belt.deleter
    def seat_belt(self):
        del self._seat_belt

    @property
    def engine_status(self):
        return self._engine_status

    @engine_status.setter
    def engine_status(self, value):
        self._engine_status = value

    @engine_status.deleter
    def engine_status(self):
        del self._engine_status

    @property
    def turn_signal(self):
        return self._turn_signal

    @turn_signal.setter
    def turn_signal(self, value):
        self._turn_signal = value

    @turn_signal.deleter
    def turn_signal(self):
        del self._turn_signal

    @property
    def parking_brake(self):
        return self._parking_brake

    @parking_brake.setter
    def parking_brake(self, value):
        self._parking_brake = value

    @parking_brake.deleter
    def parking_brake(self):
        del self._parking_brake

    @property
    def differential(self):
        return self._differential

    @differential.setter
    def differential(self, value):
        self._differential = value

    @differential.deleter
    def differential(self):
        del self._differential

    @property
    def tandem_axle_lift(self):
        return self._tandem_axle_lift

    @tandem_axle_lift.setter
    def tandem_axle_lift(self, value):
        self._tandem_axle_lift = value

    @tandem_axle_lift.deleter
    def tandem_axle_lift(self):
        del self._tandem_axle_lift

    @property
    def brake(self):
        return self._brake

    @brake.setter
    def brake(self, value):
        self._brake = value

    @brake.deleter
    def brake(self):
        del self._brake

    @property
    def light(self):
        return self._light

    @light.setter
    def light(self, value):
        self._light = value

    @light.deleter
    def light(self):
        del self._light

    @property
    def odometer(self):
        return self._odometer

    @odometer.setter
    def odometer(self, value):
        self._odometer = value

    @odometer.deleter
    def odometer(self):
        del self._odometer

    @property
    def fuel_gauge(self):
        return self._fuel_gauge

    @fuel_gauge.setter
    def fuel_gauge(self, value):
        self._fuel_gauge = value

    @fuel_gauge.deleter
    def fuel_gauge(self):
        del self._fuel_gauge

    @property
    def speed_limit(self):
        return self._speed_limit

    @speed_limit.setter
    def speed_limit(self, value):
        self._speed_limit = value

    @speed_limit.deleter
    def speed_limit(self):
        del self._speed_limit

    @property
    def navigation(self):
        return self._navigation

    @navigation.setter
    def navigation(self, value):
        self._navigation = value

    @navigation.deleter
    def navigation(self):
        del self._navigation

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @position.deleter
    def position(self):
        del self._position

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, value):
        self._period = value

    @period.deleter
    def period(self):
        del self._period
