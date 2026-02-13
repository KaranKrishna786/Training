def print_stars(count: int = 20):
    """
    Prints a line of stars to the console.

    Args:
        count (int): The number of stars to print. Defaults to 20.
                     If count is negative, it will print 0 stars.
    """
    if count < 0:
        count = 0
    print("*" * count)

if __name__ == "__main__":
    print("--- Star Printer Script ---")

    # Print a line of 15 stars
    print("\nPrinting 15 stars:")
    print_stars(15)

    # Print a line of 40 stars
    print("\nPrinting 40 stars:")
    print_stars(40)

    # Print the default number of stars (20)
    print("\nPrinting default number of stars:")
    print_stars()

    # Print 0 stars
    print("\nPrinting 0 stars:")
    print_stars(0)

    # Demonstrate handling of negative input (will print 0 stars)
    print("\nPrinting -5 stars (will print 0):")
    print_stars(-5)

    print("\n--- Script Finished ---")
