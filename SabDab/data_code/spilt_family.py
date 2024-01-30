def partition_into_groups(nums, num_groups):
    # 计算每个组的目标元素和
    target_sum = sum(nums) // num_groups
    
    # 初始化组和当前组元素和
    groups = [[] for _ in range(num_groups)]
    current_group_sum = [0] * num_groups

    # 将元素按降序排列
    sorted_indices = sorted(range(len(nums)), key=lambda i: nums[i], reverse=True)

    # 贪心地将元素放入组中
    for index in sorted_indices:
        # 选择当前组中元素和最小的组
        current_group = min(range(num_groups), key=lambda i: current_group_sum[i])

        # 将元素放入当前组
        groups[current_group].append(index)
        
        # 更新当前组元素和
        current_group_sum[current_group] += nums[index]

    return groups

# 示例，假设 nums 为包含100个整数值的列表
nums = [73, 82, 50, 47, 63, 67, 90, 54, 87, 89, 43, 72, 68, 66, 58, 77, 96, 51, 75, 64, 81, 65, 74, 61, 79, 57, 93, 60, 52, 94, 49, 80, 91, 62, 86, 78, 48, 92, 46, 69, 56, 55, 84, 76, 71, 88, 70, 85, 83, 53, 59, 45, 44, 95, 98, 97, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]

# 将 nums 分成 5 组
result_groups = partition_into_groups(nums, 5)

# 打印结果
for group_index, group in enumerate(result_groups):
    print(f"Group {group_index + 1}: {group}")
