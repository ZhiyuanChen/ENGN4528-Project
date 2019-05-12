from enum import Enum, unique


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
    Habour = 8
    Train = 9
    Toll = 10
    Fuel = 11

# Engine status: 0 Not Started, 1 Electric Started, 2 Engine Started
# Turn signal: 0 Off, 1 Left, 2 Right, 3 Emergency
# Parking Brake
# Brake
# Light
# Odometer
# Fuel gauge


class Truck(object):
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
