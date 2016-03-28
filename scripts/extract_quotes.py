#!/usr/bin/env python3
import sys


def main(path):
    quote = False

    with open(path, 'r') as fobj:
        while True:
            chr_ = fobj.read(1)
            if not chr_:
                break

            if chr_ == '"':
                if quote:
                    print(' ', end="")
                quote = not quote
            elif quote:
                print(chr_, end="")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error: Invalid Arguments!\n")
        print("Usage:")
        print("\t{} <path>".format(sys.argv[0]))
        sys.exit(1)

    main(sys.argv[1])

