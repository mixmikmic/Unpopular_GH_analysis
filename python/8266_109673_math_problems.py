import math
def is_power_of_three(n):
    """
    :type n: int
    :rtype: bool
    """
    if n < 1:
        return False
    if n == 1:
        return True
    else:
        while n > 3:
            n = n / float(3)

        if n == 3:
            return True
        else:
            return False

print is_power_of_three(1) == True
print is_power_of_three(2) == False
print is_power_of_three(3) == True
print is_power_of_three(6) == False
print is_power_of_three(9) == True
print is_power_of_three(27) == True
print is_power_of_three(28) == False
print is_power_of_three(81) == True
print is_power_of_three(100) == False
print is_power_of_three(81*3) == True

## max number



