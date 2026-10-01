package ferrota

// MAType selects the moving average used by indicators that take a matype
// parameter. Values 0-6 match TA-Lib's TA_MAType; 7 is T3 here (TA-Lib's 7 is
// MAMA, which is available as Mama instead) and 8 is an alias of T3.
type MAType int32

const (
	MATypeSMA   MAType = 0
	MATypeEMA   MAType = 1
	MATypeWMA   MAType = 2
	MATypeDEMA  MAType = 3
	MATypeTEMA  MAType = 4
	MATypeTRIMA MAType = 5
	MATypeKAMA  MAType = 6
	MATypeT3    MAType = 7
)
