# SWUFE-GeoText


* [背景知识](#background)
  * [时空轨迹数据概述](#overview)
  * [常见时空轨迹数据](#common_trajectory_data)
  * [时空轨迹数据的表示与定义](#representation_and_definition)
* [时空轨迹语义理解学习框架](#learning_framework)
  * [时空图神经网络](#spatial_temporal_graph_neural_networks)
  * [概率深度学习](#probability_deep_learning)
  * [基于底座模型的学习](#foundation_model_based_learning)
* [时空轨迹语义理解核心任务](#core_tasks)
  * [时空要素表示学习](#element_representation_learning)
    * [深度表示学习](#deep_representation_learning)
    * [解耦表示学习](#disentangled_representation_learning)
    * [因果表示学习](#causal_representation_learning)
  * [基于时空语义理解的轨迹相似学习](#trajectory_similarity_learning)
    * [时空轨迹聚类](#trajectory_clustering)
    * [时空轨迹分类](#trajectory_classification)
    * [异常行为识别](#abnormal_identification)
  * [基于时空语义理解的轨迹预测学习](#trajectory_prediction_learning)
    * [时空轨迹预测](#space_time_trajectory_prediction)
    * [时空事件预测](#space_time_event_prediction)
    * [时空数据补全](#spatio_temporal_data_completion)
* [新智能时代的时空轨迹语义理解的挑战与机遇](#challenge_and_opportunities)
  * [多模态数据处理](#multimodal_data_processing)
  * [模型的透明可释](#transparent_and_releasable)
  * [开放语义建模](#open_semantic_modeling)
  * [模型可用与资源消耗](#availability_and_resource_consumption)
  * [隐私、伦理与信任问题](#trust_issues)

<h2 id="background">背景知识</h2> 
<h3 id="overview">时空轨迹数据概述</h3> 1-7

* Alam M M, Torgo L, Bifet A. 2022. [A survey on spatio-temporal data analytics systems](https://dl.acm.org/doi/abs/10.1145/3507904). *ACM Computing Surveys 2022*.
* Erwig, Martin, et al. 1999. [Spatio-temporal data types: An approach to modeling and querying moving objects in databases](https://link.springer.com/article/10.1023/A:1009805532638). *GeoInformatica 1999*.
* Pelekis, Nikos, et al. 2004. [Literature review of spatio-temporal database models](https://www.cambridge.org/core/journals/knowledge-engineering-review/article/abs/literature-review-of-spatiotemporal-database-models/38175D21635346C9002C3C2DEDF9232D). *The Knowledge Engineering Review 2004*.
* Zhang, Junbo, et al. 2016. [DNN-based prediction model for spatio-temporal data](https://dl.acm.org/doi/abs/10.1145/2996913.2997016). *Proceedings of the 24th ACM SIGSPATIAL international conference on advances in geographic information systems 2016*.

<h3 id="common_trajectory_data">常见时空轨迹数据</h3> 8.9

* Atluri, Gowtham, Anuj Karpatne, and Vipin Kumar. 2018. [Spatio-temporal data mining: A survey of problems and methods](https://dl.acm.org/doi/abs/10.1145/3161602). *ACM Computing Surveys (CSUR) 2018*.
* Evans, Michael R., et al. 2019. [Enabling spatial big data via CyberGIS: Challenges and opportunities](https://link.springer.com/chapter/10.1007/978-94-024-1531-5_8). *CyberGIS for geospatial discovery and innovation* 2019.

<h3 id="representation_and_definition">时空轨迹数据的表示与定义</h3> 10,11

<h2 id="learning_framework">时空轨迹语义理解学习框架</h2> 12,13

* Chen W, Liang Y, Zhu Y, et al. 2024. [Deep Learning for Trajectory Data Management and Mining: A Survey and Beyond](https://arxiv.org/abs/2403.14151). *arXiv preprint 2024*.

<h3 id="deep_sequence_learning">时空图神经网络（Spatial-Temporal Graph Neural Networks）</h3> 14-18

* Goodfellow I J, Pouget-Abadie J, Mirza M, Xu B, Warde-Farley D, Ozair S, et al. 2014. [Generative adversarial nets](). In: *Proceedings of the 27th International Conference on Neural Information Processing Systems. Montreal, Canada: NIPS, 2014*.
* Chen X, Duan Y, Houthooft R, et al. 2016. [Infogan: Interpretable representation learning by information maximizing generative adversarial nets](https://proceedings.neurips.cc/paper_files/paper/2016/hash/7c9d0b1f96aebd7b5eca8c3edaa19ebb-Abstract.html). *Advances in neural information processing systems 2016*.
* Radford A, Metz L, Chintala S. 2015. [Unsupervised representation learning with deep convolutional generative adversarial networks](https://arxiv.org/abs/1511.06434). *arXiv preprint 2015*. 
* Martin Arjovsky, Soumith Chintala, Léon Bottou. 2017. [Wasserstein GAN](https://arxiv.org/abs/1701.07875). *arXiv preprint 2017*.
* Li C, Xu T, Zhu J, et al. 2017. [Triple generative adversarial nets](https://proceedings.neurips.cc/paper/2017/hash/86e78499eeb33fb9cac16b7555b50767-Abstract.html). *Advances in neural information processing systems 2017*.

<h3 id="probabilistic_deep_learning">概率深度学习</h3> 19-36

* Kingma D P, Welling M. 2013. [Auto-encoding variational bayes](https://arxiv.org/abs/1312.6114). *arXiv preprint 2013*.
* Ramchandran S, Tikhonov G, Lönnroth O, et al. 2024. [Learning conditional variational autoencoders with missing covariates](https://www.sciencedirect.com/science/article/pii/S0031320323008105). *Pattern Recognition 2024*. 
* FactorVAE
* β-VAE
* Sohl-Dickstein J, Weiss E, Maheswaranathan N, et al. 2015. [Deep unsupervised learning using nonequilibrium thermodynamics](https://proceedings.mlr.press/v37/sohl-dickstein15.html). *International conference on machine learning 2015*.
* Ho J, Jain A, Abbeel P. 2020. [Denoising diffusion probabilistic models](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html). *Advances in neural information processing systems 2020*.
* Song Y, Sohl-Dickstein J, Kingma D P, et al. 2020. [Score-based generative modeling through stochastic differential equations](https://arxiv.org/abs/2011.13456). *arXiv preprint 2020*. 
* Lin L, Li Z, Li R, et al. 2023. [Diffusion models for time-series applications: a survey](https://link.springer.com/article/10.1631/FITEE.2300310). *Frontiers of Information Technology & Electronic Engineering 2023*.
* Lim H, Kim M, Park S, et al. 2023. [Regular time-series generation using sgm](https://arxiv.org/abs/2301.08518). *arXiv preprint 2023*. 
* Li Y, Lu X, Wang Y, et al. 2022. [Generative time series forecasting with diffusion, denoise, and disentanglement](). *Advances in Neural Information Processing Systems 2022*. 
* Zhu Y, Yu J J, Zhao X, et al. 2024. [Controltraj: Controllable trajectory generation with topology-constrained diffusion model](https://arxiv.org/abs/2404.15380). *arXiv preprint 2024*. 
* Zhu Y, Ye Y, Zhang S, et al. 2024. [Difftraj: Generating gps trajectory with diffusion probabilistic model](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cd9b4a28fb9eebe0430c3312a4898a41-Abstract-Conference.html). Advances in *Neural Information Processing Systems 2024*.
* Wei T, Lin Y, Guo S, et al. 2024. [Diff-RNTraj: A Structure-aware Diffusion Model for Road Network-constrained Trajectory Generation](https://arxiv.org/abs/2402.07369). *arXiv preprint 2024*. 
* Larsen A B L, Sønderby S K, Larochelle H, et al. 2016. [Autoencoding beyond pixels using a learned similarity metric](https://proceedings.mlr.press/v48/larsen16). *International conference on machine learning 2016*. 
* Mescheder L, Nowozin S, Geiger A. 2017. [Adversarial variational bayes: Unifying variational autoencoders and generative adversarial networks](). *International conference on machine learning. PMLR, 2017*.
* Makhzani A, Shlens J, Jaitly N, et al. 2015. [Adversarial autoencoders](https://arxiv.org/abs/1511.05644). *arXiv preprint 2015*.

<h3 id="foundation_model_based_learning">基于底座模型的学习（Foundation Model-based Learning）</h3> 37-43

<h2 id="core_tasks">时空轨迹语义理解核心任务</h2> 

<h3 id="element_representation_learning">时空要素表示学习</h3> 44

<h4 id="deep_representation_learning">深度表示学习（Deep Representation Learning）</h3>  45-72

<h4 id="disentangled_representation_learning">解耦表示学习（Disentangled Representation Learning）</h3> 73-86

<h4 id="causal_representation_learning">因果表示学习（Disentangled Representation Learning）</h3> 87-92

<h3 id="trajectory_similarity_learning">基于时空语义理解的轨迹相似学习</h2> 

<h4 id="trajectory_clustering">时空轨迹聚类</h4> 93-106

<h4 id="trajectory_classification">时空轨迹分类</h4> 107-121

<h4 id="abnormal_identification">异常行为识别</h4> 122-133


<h3 id="trajectory_prediction_learning">基于时空语义理解的轨迹预测学习</h3> 

<h4 id="space_time_trajectory_prediction">时空轨迹预测</h4> 134-174

* 个推1-5
* 以下为当前文中34、37、40、36、38、39、41、42
* Dai S, Yu Y, Fan H, et al. 2022. [Spatio-temporal representation learning with social tie for personalized poi recommendation](https://link.springer.com/article/10.1007/s41019-022-00180-w). *Data Science and Engineering, 2022*.
* Li M, Zheng W, Xiao Y, et al. 2020. [An Adaptive POI Recommendation Algorithm by Integrating User's Temporal and Spatial Features in LBSNs](https://dl.acm.org/doi/abs/10.1145/3414274.3414494). *Proceedings of the 3rd International conference on data science and information technology. 2020*.
* Tahmasbi H, Jalali M, Shakeri H. 2021. [Modeling user preference dynamics with coupled tensor factorization for social media recommendation](https://link.springer.com/article/10.1007/s12652-020-02714-4). *Journal of Ambient Intelligence and Humanized Computing, 2021*.
* Wang X, Sun G, Fang X, et al. 2022. [Modeling spatio-temporal neighbourhood for personalized point-of-interest recommendation](https://www.ijcai.org/proceedings/2022/490). *Proceedings of IJCAI. 2022*.
* Parveen R, Varma N S. 2021. [Friend's recommendation on social media using different algorithms of machine learning](https://www.sciencedirect.com/science/article/pii/S2666285X21000406). *Global Transitions Proceedings, 2021*.
* Saraswathi K, Mohanraj V, Suresh Y, et al. [Deep Learning Enabled Social Media Recommendation Based on User Comments](https://cdn.techscience.cn/ueditor/files/csse/TSP_CSSE-44-2/TSP_CSSE_27987/TSP_CSSE_27987.pdf). *Computer Systems Science & Engineering, 2023*.
* Liu H, Tong Y, Han J, et al. 2020. [Incorporating multi-source urban data for personalized and context-aware multi-modal transportation recommendation](https://ieeexplore.ieee.org/abstract/document/9063461/). *IEEE Transactions on Knowledge and Data Engineering, 2020*.
* Zhu M, Hu J, Pu Z, et al. 2019. [Personalized Context-Aware Multi-Modal Transportation Recommendation](https://arxiv.org/abs/1910.12601). *arXiv preprint 2019*.

<h4 id="space_time_event_prediction">时空事件预测</h4> 175-202

* 城感1
* 以下为当前文中43、44、46、47
* Yang G, Yu H, Xi H. 2022. [A Spatio-Temporal Traffic Flow Prediction Method Based on Dynamic Graph Convolution Network](https://ieeexplore.ieee.org/abstract/document/10033842). *2022 34th Chinese Control and Decision Conference (CCDC). IEEE, 2022*.
* Cheng M, Jiang G P, Song Y, et al. 2022. [Dynamic Spatio-temporal traffic flow prediction based on multi fusion graph attention network](https://ieeexplore.ieee.org/abstract/document/9902776). *2022 41st Chinese Control Conference (CCC). IEEE, 2022*.
* Amato F, Guignard F, Robert S, et al. 2020. [A novel framework for spatio-temporal prediction of environmental data using deep learning](https://www.nature.com/articles/s41598-020-79148-7). *Scientific reports, 2020*.
* Li T, Zhang J, Bao K, et al. 2020. [Autost: Efficient neural architecture search for spatio-temporal prediction](https://dl.acm.org/doi/abs/10.1145/3394486.3403122). *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020*.

<h4 id="spatio_temporal_data_completion">时空数据补全</h4> 203-214

* 48、补全、补全4
* Chen J, Chen P. 2017. [A method based on tensor decomposition for missing multi-dimensional data completion](https://ieeexplore.ieee.org/abstract/document/8078795/). *2017 IEEE 2nd International Conference on Big Data Analysis (ICBDA). IEEE, 2017*.
* Kong X, Zhou W, Shen G, et al. 2023. [Dynamic graph convolutional recurrent imputation network for spatiotemporal traffic missing data](https://www.sciencedirect.com/science/article/pii/S0950705122012849). *Knowledge-Based Systems, 2023*.
* Cai L, Sha C, He J, et al. 2023. [Spatial–Temporal Data Imputation Model of Traffic Passenger Flow Based on Grid Division](https://www.mdpi.com/2220-9964/12/1/13). *ISPRS International Journal of Geo-Information, 2023*.

[新智能时代的时空轨迹语义理解的挑战与机遇](#challenge_and_opportunities)
  * [多模态数据处理](#multimodal_data_processing)
  * [模型的透明可释](#transparent_and_releasable)
  * [开放语义建模](#open_semantic_modeling)
  * [模型可用与资源消耗](#availability_and_resource_consumption)
  * [隐私、伦理与信任问题](#trust_issues)

<h2 id="challenge_and_opportunities">新智能时代的时空轨迹语义理解的挑战与机遇</h2>

<h3 id="multimodal_data_processing">多模态数据处理</h3>

<h3 id="transparent_and_releasable">模型的透明可释</h3>

<h3 id="open_semantic_modeling">开放语义建模</h3>

<h3 id="availability_and_resource_consumption">模型可用与资源消耗</h3>

<h3 id="trust_issues">伦理与信任问题</h3>



