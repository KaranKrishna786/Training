import sys

def add_numbers(num1, num2):
    """Adds two numbers and returns the result."""
    return num1 + num2

def main():
    """
    This script takes two numbers as command-line arguments and prints their sum.
    """
    if len(sys.argv) != 3:
        print("Usage: python add_script.py <number1> <number2>")
        sys.exit(1)

    try:
        number1 = float(sys.argv[1])
        number2 = float(sys.argv[2])
    except ValueError:
        print("Error: Both arguments must be valid numbers.")
        sys.exit(1)

    result = add_numbers(number1, number2)
    print(f"The sum of {number1} and {number2} is: {result}")

if __name__ == "__main__":
    main()
