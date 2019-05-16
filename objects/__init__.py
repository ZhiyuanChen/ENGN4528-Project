import cv2
import base64
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
        self.message = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('ascii')


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
    # Parking Brake
    # Brake
    # Light
    # Odometer
    # Fuel gauge
    def __init__(self, engine_status=0, turn_signal=0, parking_brake=0, brake=0, light=0, odometer=0, fuel_gauge=0,
                 speed_limit=0, navigation=0, position=Position.NotSet, period=Period.NotSet):
        self._EngineStatus = engine_status
        self._TurnSignal = turn_signal
        self._ParkingBrake = parking_brake
        self._Brake = brake
        self._Light = light
        self._Odometer = odometer
        self._FuelGauge = fuel_gauge
        self._SpeedLimit = speed_limit
        self._Navigation = navigation
        self._Position = position
        self._Period = period

    @property
    def EngineStatus(self):
        return self._EngineStatus

    @EngineStatus.setter
    def EngineStatus(self, value):
        self._EngineStatus = value

    @EngineStatus.deleter
    def EngineStatus(self):
        del self._EngineStatus

    @property
    def TurnSignal(self):
        return self._TurnSignal

    @TurnSignal.setter
    def TurnSignal(self, value):
        self._TurnSignal = value

    @TurnSignal.deleter
    def TurnSignal(self):
        del self._TurnSignal

    @property
    def ParkingBrake(self):
        return self._ParkingBrake

    @ParkingBrake.setter
    def ParkingBrake(self, value):
        self._ParkingBrake = value

    @ParkingBrake.deleter
    def ParkingBrake(self):
        del self._ParkingBrake

    @property
    def Brake(self):
        return self._Brake

    @Brake.setter
    def Brake(self, value):
        self._Brake = value

    @Brake.deleter
    def Brake(self):
        del self._Brake

    @property
    def Light(self):
        return self._Light

    @Light.setter
    def Light(self, value):
        self._Light = value

    @Light.deleter
    def Light(self):
        del self._Light

    @property
    def Odometer(self):
        return self._Odometer

    @Odometer.setter
    def Odometer(self, value):
        self._Odometer = value

    @Odometer.deleter
    def Odometer(self):
        del self._Odometer

    @property
    def FuelGauge(self):
        return self._FuelGauge

    @FuelGauge.setter
    def FuelGauge(self, value):
        self._FuelGauge = value

    @FuelGauge.deleter
    def FuelGauge(self):
        del self._FuelGauge

    @property
    def SpeedLimit(self):
        return self._SpeedLimit

    @SpeedLimit.setter
    def SpeedLimit(self, value):
        self._SpeedLimit = value

    @SpeedLimit.deleter
    def SpeedLimit(self):
        del self._SpeedLimit

    @property
    def Navigation(self):
        return self._Navigation

    @Navigation.setter
    def Navigation(self, value):
        self._Navigation = value

    @Navigation.deleter
    def Navigation(self):
        del self._Navigation

    @property
    def Position(self):
        return self._Position

    @Position.setter
    def Position(self, value):
        self._Position = value

    @Position.deleter
    def Position(self):
        del self._Position

    @property
    def Period(self):
        return self._Period

    @Period.setter
    def Period(self, value):
        self._Period = value

    @Period.deleter
    def Period(self):
        del self._Period

