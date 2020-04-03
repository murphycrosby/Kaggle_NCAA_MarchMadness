import argparse
from datetime import datetime
from datetime import timedelta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts date to daynum.')
    parser.add_argument('--sub', action='store_true')
    parser.add_argument('--add', action='store_true')
    parser.add_argument('--date', help='date to convert')
    parser.add_argument('--numdays')
    parser.add_argument('--startdate', help='start date to count from', default='11/5/2019')
    args = parser.parse_args()

    start_date = datetime.strptime(args.startdate, '%m/%d/%Y')
    if args.sub is True and args.date is not None:
        day_of_year = datetime.strptime(args.date, '%m/%d/%Y')
        print(day_of_year - start_date)

    if args.add is True and args.numdays is not None:
        print(start_date + timedelta(days=int(args.numdays)))
