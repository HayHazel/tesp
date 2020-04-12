declare -r WINTER_START="2013-01-03 00:00:00"
declare -r WINTER_END="2013-01-05 00:00:00"
declare -r SUMMER_START="2013-08-01 00:00:00"
declare -r SUMMER_END="2013-08-03 00:00:00"
declare -r ROOT=$1
declare -r STEPS="4"

./run_ems_case.sh "$ROOT" "$WINTER_START" "$WINTER_END" "Winter$ROOT" "$STEPS"
./run_ems_case.sh "$ROOT" "$SUMMER_START" "$SUMMER_END" "Summer$ROOT" "$STEPS"

