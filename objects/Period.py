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
