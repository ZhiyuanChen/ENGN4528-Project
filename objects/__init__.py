from enum import Enum, unique


class Image(object):
    def __init__(self, image):
        self.image = image
        self.windshield = image[0:490, 0:1200]
        self.left_mirror = image[58:296, 16:193]
        self.right_mirror = image[58:296, 1082:1259]
        self.navigation = image[578:683, 246:424]
        self.assistant = image[268:472, 27:287]
        self.odometer = image[653:662, 626:680]
        self.fuel_gauge = image[612:620, 920:974]
        self.left_turn = image[565:584, 735:756]
        self.right_turn = image[565:584, 863:883]
        self.parking_break = image[565:584, 901:924]
        self.seat_belt = image[565:585, 927:943]
        self.battery_charge = image[578:581, 969:982]
        self.malfunction_indicator = image[679:694, 907:929]
        self.glow_plug = image[680:694, 907:929]
        self.light0 = image[567:575, 1108:1120]
        self.light1 = image[569:577, 1123:1135]
        self.light2 = image[570:578, 1138:1150]
        self.light3 = image[570:578, 1154:1170]
        self.light4 = image[572:578, 1174:1179]
        self.light5 = image[579:583, 1174:1179]
        self.failure0 = image[684:693, 624:641]
        self.failure1 = image[682:694, 647:665]
        self.failure2 = image[682:693, 670:688]
        self.failure3 = image[682:693, 693:712]

    @staticmethod
    def is_green(part):
        return True if part[1].mean() > 10 else False

    @staticmethod
    def is_red(part):
        return True if part[2].mean() > 10 else False

    @staticmethod
    def is_yellow(part):
        return True if (part[1] + part[2]).mean() > 10 else False


@unique
class period(Enum):
    NotSet = 0
    Spring = 1
    Morning = 2
    Noon = 3
    Afternoon = 4
    Fall = 5
    Evening = 6


@unique
class position(Enum):
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
    def __init__(self, engine_status=0, turn_signal=0, parking_brake=0, brake=0, light=0, odometer=0, fuel_gauge=0,
                 speed_limit=0, navigation=0, position=position.NotSet, period=period.NotSet):
        self._engine_status = engine_status
        self._turn_signal = turn_signal
        self._parking_brake = parking_brake
        self._brake = brake
        self._light = light
        self._odometer = odometer
        self._fuel_gauge = fuel_gauge
        self._speed_limit = speed_limit
        self._navigation = navigation
        self._position = position
        self._period = period

    def init(self, image):
        image = Image(image)
        if image.is_green(image.left_turn):
            if image.is_green(image.right_turn):
                self.turn_signal = 3
            else:
                self.turn_signal = 1
        elif image.is_green(image.right_turn):
            self.turn_signal = 2
        else:
            self.turn_signal = 0
        self.parking_brake = 1 if image.is_red(image.parking_break) else 0

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
