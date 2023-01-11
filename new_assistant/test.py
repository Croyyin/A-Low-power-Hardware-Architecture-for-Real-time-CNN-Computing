
def find_first_neg(input_list):
    result = 0
    for i in input_list:
        if i < 0:
            result = i
            break
    return result


def alg_1(order_list, queue_len):
    start = 0
    end = queue_len
    result_list = []
    while end <= len(order_list):
        new_list = order_list[start : end]
        current_result = find_first_neg(new_list)
        result_list.append(current_result)
        start += 1
        end += 1
    return result_list

def find_first_neg_pos(input_list):
    result = -1
    for i, l in enumerate(input_list):
        if l < 0:
            result = i
            break
    return result

def alg_2(order_list, queue_len):
    start = 0
    end = queue_len
    result_list = []
    first_neg_pos = -1
    while end <= len(order_list):
        if first_neg_pos < end and first_neg_pos >= start:
            current_result = order_list[first_neg_pos]
        else:
            new_list = order_list[start : end]
            first_neg_pos = find_first_neg_pos(new_list)
            if first_neg_pos == -1:
                current_result = 0
            else:
                first_neg_pos += start
                current_result = order_list[first_neg_pos]

        result_list.append(current_result)
        start += 1
        end += 1
    return result_list



def test1(str1, str2, step_n):
    if str1 == str2:
        return step_n - 2
    if str1 != str2 and step_n > len(str1):
        return step_n
    times_list = []
    for i in range(len(str1) - step_n + 1):
        head = str1[:i]
        rear = str1[i+step_n:]
        mid = str1[i:i+step_n][::-1]
        str1 = head + mid + rear
        current_times = test1(str1, str2, step_n + 1)
        times_list.append(current_times)
    result = min(times_list)
    return result


def test2(str1, str2, step_n):
    if str1 == str2:
        return step_n - 2
    if str1 != str2 and step_n > len(str1):
        return step_n
    times_list = []
    for i in range(len(str1) - step_n + 1):
        head = str1[:i]
        rear = str1[i+step_n:]
        mid = str1[i:i+step_n][::-1]
        str1 = head + mid + rear
        current_times = test1(str1, str2, step_n + 1)
        times_list.append(current_times)
        if current_times + 2 == step_n:
            break
    result = min(times_list)
    return result


def test3(s):
    new_str = ""
    str_len = len(s)
    i = 0
    while i < str_len:
        if s[i] == "-" and s[i + 1] == new_str[-1]:
            i += 1
        elif s[i] == "-" and s[i + 1] != new_str[-1]:
            head = ord(new_str[-1])
            rear = ord(s[i + 1])

            if head < rear:
                for j in range(head + 1, rear):
                    new_str += chr(j) 
            else:
                for j in range(head - 1, rear, -1):
                    new_str += chr(j) 
        else:
            new_str += s[i]
        i += 1
    return new_str


def nums_to_list(nums):
    list1 = []
    for n in nums:
        list1.append(int(n))
    return list1

def cpt_OP(file_name):
    kernel_size_col = []
    kernel_stride_col = []
    last_in_channel = []
    in_height = []
    out_channel = []
    out_height = []
    pre_in_channel = []

    layer_result = []
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:

            if "kernel_size_col" in line and "pooling" not in line:
                kernel_size_col = nums_to_list(line.split(":")[-1].split(","))
            if "kernel_stride_col" in line and "pooling" not in line:
                kernel_stride_col = nums_to_list(line.split(":")[-1].split(","))
            if "last_in_channel" in line:
                last_in_channel = nums_to_list(line.split(":")[-1].split(","))
            if "in_height" in line:
                in_height = nums_to_list(line.split(":")[-1].split(","))
            if "out_channel" in line:
                out_channel = nums_to_list(line.split(":")[-1].split(","))
            if "out_height" in line:
                out_height = nums_to_list(line.split(":")[-1].split(","))
            if "pre_in_channel" in line:
                pre_in_channel = nums_to_list(line.split(":")[-1].split(","))


    for i in range(len(kernel_size_col)):
        other_cpt = ((kernel_size_col[i] * 2 - 1) * pre_in_channel[i] + pre_in_channel[i]) * (kernel_size_col[i] - 1)
        last_cpt = ((kernel_size_col[i] * 2 - 1) * last_in_channel[i] + last_in_channel[i]) * (kernel_size_col[i] - 1)
        result = other_cpt + last_cpt
        layer_result.append(result)
    return layer_result
    


if __name__ == "__main__":
    # model = LeNet()
    # total = sum([param.nelement() for param in model.parameters()])
    # print(total)
    # list_1 = [3, -11, -2, 19, 37, 64, -18]
    # result = alg_2(list_1, 3)
    # print(result)
    # str1 = "10101010"
    # str2 = "00101011"
    # result = test2(str1, str2, 2)
    # if(result > len(str1)):
    #     print("No!")
    # else:
    #     print("Yes! " + str(result))
    print("test") 
    result = cpt_OP("test_model_configs/our/le/max_cycle_168.txt")
    sum_ = 0
    for i in range(len(result)):
        sum_ += result[i]
    
    sum_ /= 1000
    print(result)
    print(sum_ / 20)