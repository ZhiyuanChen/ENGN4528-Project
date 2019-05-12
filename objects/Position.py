from enum import Enum, unique


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
