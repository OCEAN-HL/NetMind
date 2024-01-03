import csv
import json
import random


def generate_data_x(computing_request, bandwidth_request):
    prob = random.random()  # Generate a random number between 0 and 1
    if prob < 1 / 4:  # uRLLC
        # If the number is less than 0.5, generate a random number between 10 and 20
        return [
            0,  # service_type
            random.randrange(13, 20) / 25,  # fronthal_delay
            random.randint(30, 40) / 40,  # end-to-end delay
            random.randint(15, computing_request) / computing_request,  # DU
            random.randint(5, 10) / computing_request,  # CU-UP
            random.randint(5, 10) / computing_request,  # CU-CP
            random.randint(15, computing_request) / computing_request,  # UPF
            random.randint(2, 4) / bandwidth_request,  # bandwidth
        ]
    elif 1 / 4 <= prob < 2 / 4:  # eMBB
        return [
            1 / 3,  # service_type
            random.randrange(20, 25) / 25,  # fronthal_delay
            random.randint(30, 40) / 40,  # midhaul delay
            random.randint(15, computing_request) / computing_request,  # DU
            random.randint(15, computing_request) / computing_request,  # CU-UP
            random.randint(5, 10) / computing_request,  # CU-CP
            0,  # UPF
            random.randint(4, bandwidth_request) / bandwidth_request,  # bandwidth
        ]
    elif 2 / 4 <= prob < 3 / 4:  # mMTC
        return [
            2 / 3,  # service_type
            random.randrange(20, 25) / 25,  # fronthal_delay
            random.randint(30, 40) / 40,  # midhaul delay
            random.randint(15, computing_request) / computing_request,  # DU
            random.randint(5, 10) / computing_request,  # CU-UP
            random.randint(15, computing_request) / computing_request,  # CU-CP
            0,  # UPF
            random.randint(4, bandwidth_request) / bandwidth_request,  # bandwidth
        ]
    elif 3 / 4 < prob:  # no requests
        return [
            3 / 3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]


computing_resource = 120
bandwidth_resource = 35
bandwidth_request = 7
computing_request = 25
node_number = 4 + 1
row_data = []
for i in range(6000): 
    source = random.randrange(1, node_number) / (node_number - 1)

    request_1 = generate_data_x(computing_request, bandwidth_request)
    request_1.append(1)  
    request_1.append(source)  
    request_1.append(random.randrange(computing_resource - 30, computing_resource) / computing_resource)  

    request_2 = generate_data_x(computing_request, bandwidth_request)
    request_2.append(0)
    request_2.append(source)  
    request_2.append(random.randrange(computing_resource - 30, computing_resource) / computing_resource)  

    request_3 = generate_data_x(computing_request, bandwidth_request)
    request_3.append(1)
    request_3.append(source)  
    request_3.append(random.randrange(computing_resource - 30, computing_resource) / computing_resource)  

    request_4 = generate_data_x(computing_request, bandwidth_request)
    request_4.append(0)
    request_4.append(source)  
    request_4.append(random.randrange(computing_resource - 30, computing_resource) / computing_resource)  

    request_5 = [
        3 / 3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    request_5.append(1)  
    request_5.append(source)  
    request_5.append(2)  

    # 当前位置信息
    lists = [request_1, request_2, request_3, request_4, request_5]

    chosen_list = random.choice(lists)
    chosen_list.append(1)

    for lst in lists:
        if lst is not chosen_list:
            lst.append(0)

    data_sample = (
        [request_1, request_2, request_3, request_4, request_5],
        [[0, 1, 2, 3, 1, 2], [1, 2, 3, 0, 4, 4]],
        [
            [11, random.randrange(bandwidth_resource - 5, bandwidth_resource) / bandwidth_resource],
            [12, random.randrange(bandwidth_resource - 5, bandwidth_resource) / bandwidth_resource],
            [7, random.randrange(bandwidth_resource - 5, bandwidth_resource) / bandwidth_resource],
            [10, random.randrange(bandwidth_resource - 5, bandwidth_resource) / bandwidth_resource],
            [20, 3],
            [20, 3],
        ],
    )

    row_data.append(data_sample)


with open("/code/src/SFC/CODER/EDcode/Network_1/data_1.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for row in row_data:
        json_str = json.dumps(row)
        writer.writerow([json_str])


