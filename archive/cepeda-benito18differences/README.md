The research is based on NSDUH dataset, which has versioning issues.
The survey is concluded each year, after which all the responds (including the ones from the previous years) are collected and published. It seems that when the survey format changes, the researchers also change all the previous years’ answers, which leads to each year’s survey has multiple versions which differ.

Because of that, it’s hard to pick the exact combination of datasets’ versions such that the total number of records would match one one reported in the paper.

Also, it seem problematic to aquire all versions for all years’ surveys, as we’ve only found the last versions for each year. However, there are also full csv files for each year, which includes all the answers from the previous years. The idea was that it was possible to split those into multiple versions for multiple years.

As the datasets are pretty large, I created an efficient rust tool which loops trough all the datasets, collects the necessary data, splits the datasets per year and does additional things. Using that, I could check all the combinations of datasets’ versions.

The paper reports the number of rows for people >= 18 years and no combination comes closer to that number than +-300 records.

Nevertheless I decided that it's "close enough" to the paper so I move on, and the next step would be to categorize records into urban and rural.

The paper cites Cromartie and Parker, 2016. I assume that it's this [site](https://www.ers.usda.gov/topics/rural-economy-population/rural-classifications/what-is-rural/), since it's the only other occurrence of those surnames. All the data sources at the bottom lead to Access Denied pages, and I couldn't find related columns in the NSDUH itself (except the one which description I have attached as a screenshot), but the percentages of urban vs rural records do not match ones claimed in the paper. I found [the following](https://www.ers.usda.gov/data-products/rural-urban-continuum-codes.aspx) dataset, but couldn't find any address related info (neither did Stacy with the other paper, which we had also to abandon because of lack of the addresses), so I assume that it's a dead end as well
