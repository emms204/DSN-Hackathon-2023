import argparse
def statement(stat):
    print(stat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TESTING......")
    parser.add_argument(
        "--statement",
        help="Statement to print",
        default=
        'What sort of a grace be this'
    )
    args = parser.parse_args()
    statement(args.statement)