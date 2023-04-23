def get_mean(lst):
    total = 0
    count = 0
    for item in lst:
        if isinstance(item, int) or isinstance(item, float):
            total += item
        else:
            assert isinstance(item, list), f"If an item in a list isn't an int or float then it shold be a list: got {type(item)}"
            total2 = 0
            count2 = 0
            for item2 in item:
                assert isinstance(item2, int) or isinstance(item2, float), f"item2 must be an int or float: got {type(item2)}"
                total2 += item2
                count2 += 1
            total += (total2/count2)
        count += 1
    return total/count

if __name__ == "__main__":

    lst = [[59.55,60.89], 55.22, 49.82, 40.07, 25.27, 47.14, 59.68]

    print(get_mean(lst))

