class str_gen:
    def basic_str(self):
        b_s = ""
        params_list = [p for p in dir(self) if not p.startswith("__") and not callable(getattr(self, p))]
        for p in params_list:
            k = getattr(self, p)
            if isinstance(k, str):
                b_s += p + ":" + k
            elif isinstance(k, list):
                if isinstance(k[0], list):
                    b_s += p + "_row:"
                    for i in range(len(k)):
                        b_s += str(k[i][0]) + ","
                    b_s = b_s[:-1]
                    b_s += "\n"
                    b_s += p + "_col:"
                    for i in range(len(k)):
                        b_s += str(k[i][1]) + ","
                    b_s = b_s[:-1]
                else:
                    b_s += p + ":"
                    for i in k:
                        b_s += str(i) + ","
                    b_s = b_s[:-1]
            else:
                b_s += p + ":" + str(k)
            b_s += "\n"
        return b_s[:-1]


class test_b(str_gen):
    def __init__(self) -> None:
        super().__init__()
        self.test_string = "hahnibamser"
        self.test_olist = [1, 2, 3, 4, 5]
        self.test_tlist = [[1, 2], [1, 2], [1, 2]]



if __name__ == "__main__":

    test_c = test_b()
    test_c.new_list_two = [[2, 3], [1, 2], [3, 4]]
    result = test_c.basic_str()
    with open("./json/ast.txt", "w") as f:
        f.write(result)
    print(result)