#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Lenovo
# @Date:   2019-06-05 19:27:58
# @Last Modified by:   zy19950209
# @Last Modified time: 2019-06-05 20:00:51
def fun(strs):
    strs = list(strs)
    print(strs)
    nums = []
    i=0
    p=1
    while i<len(strs) and p<len(strs) and strs[i]!=strs[p]:
        nums.append("1")
        nums.append(strs[i])
        p=p+1
        i=i+1
        if strs[i]==strs[p] and p<len(strs) and i<len(strs):
        	j=1
        	while p<len(strs) and i<len(strs) and strs[i]==strs[p]:

        		j=j+1
        		if p==len(strs)-1:
        			nums.append(str(j))
        			nums.append(strs[i])
        			break
        		elif p<len(strs) and i<len(strs) and strs[i]!=strs[p+1]:
        			p=p+1
        			i=p
        #     i=p+1
        #     p=p+1
        #     nums.append(j)
        #     nums.append(strs[i])
        #     continue
    return nums
strs = "111221"
print(fun(strs))