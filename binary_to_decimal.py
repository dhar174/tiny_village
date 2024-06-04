# Corrected Python program to convert a binary string to a decimal number


import time


def binary_to_decimal(binary_str):
    # Initialize the result variable. This will hold the final decimal value.
    decimal_value = 0
    print("Step 1: Initialize the decimal value to 0.")
    time.sleep(2)

    # Get the length of the binary string. This is needed because the value of each bit in the binary string
    # depends on its position from the right end (or least significant bit).
    length = len(binary_str)
    print(
        f"Step 2: Get the length of the binary string '{binary_str}'. The length is {length}."
    )
    time.sleep(2)

    # Iterate over each character in the binary string. We need to process each bit in the binary string.
    for i in range(length):
        # Convert the character to an integer (0 or 1). This is necessary because the characters in the binary string
        # are initially strings, and we need to perform mathematical operations on them.
        bit = int(binary_str[i])
        # print(
        #     f"Step 3.{i+1}: Convert the character '{binary_str[i]}' to an integer. The integer is {bit}."
        # )
        time.sleep(2)

        # In a binary number, each bit (0 or 1) represents a power of 2, based on its position from the right end. This is a fundamental concept in binary number systems.
        # The rightmost bit (also known as the least significant bit) represents 2 to the power of 0. The next bit to the left represents 2 to the power of 1, and so on.
        # For example, in the binary number 1011, the rightmost '1' represents 2^0 (which is 1), the next '1' to the left represents 2^1 (which is 2), the '0' represents 2^2 (which is 4, but it's multiplied by 0 so it contributes nothing to the final value), and the leftmost '1' represents 2^3 (which is 8).
        # So, to convert a binary number to a decimal number, we need to iterate over each bit in the binary number, from right to left, and add the value of 2 raised to the corresponding power (which is the position of the bit from the right end) to a running total.
        # In this code, we're iterating over the binary string from left to right (which is more straightforward in most programming languages), so we need to adjust the power of 2 that we're calculating. We do this by subtracting the current index 'i' from the length of the binary string minus 1. This effectively reverses the position of the bit, so that the rightmost bit is considered first and the leftmost bit is considered last.
        # For example, if the binary string is '1011' (which has a length of 4), when we're processing the rightmost '1' (at index 3), we calculate the power as (4 - 1 - 3) = 0, which is correct. When we're processing the leftmost '1' (at index 0), we calculate the power as (4 - 1 - 0) = 3, which is also correct.
        # So, the expression (length - 1 - i) is calculating the correct power of 2 for the current bit, based on its position from the right end of the binary string.
        decimal_value += bit * (2 ** (length - 1 - i))
        print(
            f"Step 3.{i+1}: The current bit is {bit} and its position from the right end is {length - 1 - i}. "
            f"So, we calculate its value as {bit} * (2 ** {length - 1 - i}) = {bit * (2 ** (length - 1 - i))}. "
            f"We add this value to the running total of the decimal value. The decimal value was {decimal_value - bit * (2 ** (length - 1 - i))} and is now {decimal_value}."
        )
        time.sleep(2)

    # Return the final decimal value after all bits in the binary string have been processed.
    return decimal_value


# Test the function with a binary string. This demonstrates how to use the function.
binary_input = "00110011"
# Convert the binary string to decimal using the function.
result = binary_to_decimal(binary_input)
# Print the result. This shows the decimal equivalent of the binary string.
print(f"The decimal value of the binary string {binary_input} is: {result}")
