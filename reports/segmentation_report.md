# Economic Customer Segmentation Report

Clustered on economic variables only (historical profit, CLV, revenue, funding contribution, expected loss, operating cost, engagement, transaction volume, product count, churn probability, balance, credit utilization) -- not demographics. Segment names come from a deterministic rule over each cluster's standardized centroid (see `segmentation.profiling`); a cluster only gets one of the spec's example names if its centroid is actually extreme on the matching dimension(s).

- k selected by silhouette score: **7** (K-Means silhouette=0.2227, hierarchical cross-check on a 5,000-customer subsample=0.1403)

## k selection

The full table below includes k=3-4, which score marginally higher on raw silhouette but produce clusters too coarse to be business-actionable (a lone 'High Value' cluster plus one or two large, barely-differentiated ones -- see `clustering.MIN_BUSINESS_K`). The selected k is the silhouette-argmax restricted to k >= 5.

 k  silhouette_score    inertia
 3            0.2653 81232.4549
 4            0.1998 72369.4257
 5            0.2162 65100.6668
 6            0.2170 58380.5727
 7            0.2227 54045.3286
 8            0.2001 50568.1533
 9            0.1991 48080.5008

## Segment profiles (raw, unstandardized feature means)

                    n_customers  historical_economic_profit  total_customer_economic_value  total_revenue  total_deposit_contribution  total_expected_loss  total_operating_cost  login_frequency  transaction_volume_avg_3m  product_count  p_churn_monthly  average_balance_avg_3m  utilization
segment_name                                                                                                                                                                                                                                                                                     
Balanced Segment 0         3808                      320.91                         587.58         137.01                      319.03                11.96                121.73            16.54                     553.56           3.30             0.02                 5788.91         0.01
Credit-Driven              1185                      757.18                        1867.32        1189.22                      247.87               561.78                117.05            16.17                     616.70           4.46             0.02                 6480.83         0.65
Balanced Segment 2         2083                       45.94                         163.52          26.39                       64.61                 6.34                 38.40            11.13                     244.09           3.09             0.06                 3937.87         0.05
Deposit Funders            2524                     1021.97                        1844.04         602.06                      665.42                34.60                208.70            22.83                    1547.03           4.12             0.01                13005.10         0.04
High Value II              1144                     2717.50                        5219.28        3774.51                      603.24              1423.70                234.62            22.27                    1443.49           4.88             0.01                12048.47         0.59
Transactional              1656                      330.58                        1511.69         241.59                      182.38                37.29                 56.10            21.24                    1601.88           4.57             0.01                17733.81         0.17
High Value I                247                     5092.19                        9210.67        8995.66                      722.25              4359.91                264.86            24.16                    1814.70           5.09             0.01                15148.32         0.73

