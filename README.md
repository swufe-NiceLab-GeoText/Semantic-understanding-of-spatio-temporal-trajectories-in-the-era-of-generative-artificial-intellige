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
<h3 id="overview">时空轨迹数据概述</h3>

* Alam M M, Torgo L, Bifet A. 2022. [A survey on spatio-temporal data analytics systems](https://dl.acm.org/doi/abs/10.1145/3507904). *ACM Computing Surveys 2022*.
* Erwig, Martin, et al. 1999. [Spatio-temporal data types: An approach to modeling and querying moving objects in databases](https://link.springer.com/article/10.1023/A:1009805532638). *GeoInformatica 1999*.
* Shekhar S, Jiang Z, Ali R Y, et al. 2015. [Spatiotemporal data mining: A computational perspective](https://www.mdpi.com/2220-9964/4/4/2306). *ISPRS International Journal of Geo-Information, 2015*.
* Pelekis, Nikos, et al. 2004. [Literature review of spatio-temporal database models](https://www.cambridge.org/core/journals/knowledge-engineering-review/article/abs/literature-review-of-spatiotemporal-database-models/38175D21635346C9002C3C2DEDF9232D). *The Knowledge Engineering Review 2004*.
* Yao D, Zhang C, Huang JH, Chen YX, Bi JP. 2018. [Semantic understanding of spatio-temporal data: Technology & application](https://www.jos.org.cn/josen/article/abstract/5576). *Ruan Jian Xue Bao/Journal of Software, 2018*.
* Gao Q, Zhang FL, Wang RJ, Zhou F. 2017. [Trajectory big data: A review of key technologies in data processing](https://www.jos.org.cn/josen/article/abstract/5143). *Ruan Jian Xue Bao/Journal of Software, 2017*.
* Zhang, Junbo, et al. 2016. [DNN-based prediction model for spatio-temporal data](https://dl.acm.org/doi/abs/10.1145/2996913.2997016). *Proceedings of the 24th ACM SIGSPATIAL international conference on advances in geographic information systems 2016*.

<h3 id="common_trajectory_data">常见时空轨迹数据</h3> 

* Atluri, Gowtham, Anuj Karpatne, and Vipin Kumar. 2018. [Spatio-temporal data mining: A survey of problems and methods](https://dl.acm.org/doi/abs/10.1145/3161602). *ACM Computing Surveys (CSUR) 2018*.
* Evans, Michael R., et al. 2019. [Enabling spatial big data via CyberGIS: Challenges and opportunities](https://link.springer.com/chapter/10.1007/978-94-024-1531-5_8). *CyberGIS for geospatial discovery and innovation 2019*.

<h3 id="representation_and_definition">时空轨迹数据的表示与定义</h3> 

* ao HL, Tang HN, Wang F, Xu YJ. 2021. [Survey on trajectory representation learning techniques](https://www.jos.org.cn/josen/article/abstract/6210). *Ruan Jian Xue Bao/Journal of Software 2021*.
* Li CN, Feng GW, Yao H, Liu RY, Li YN, Xie K, Miao QG. 2024. [Survey on Trajectory Anomaly Detection](https://www.jos.org.cn/josen/article/abstract/6996). *Ruan Jian Xue Bao/Journal of Software 2024*.

<h2 id="learning_framework">时空轨迹语义理解学习框架</h2> 

* Chen W, Liang Y, Zhu Y, et al. 2024. [Deep Learning for Trajectory Data Management and Mining: A Survey and Beyond](https://arxiv.org/abs/2403.14151). *arXiv preprint 2024*.
* LECUN Y, BENGIO Y, HINTON G. 2015. [Deep learning](https://www.nature.com/articles/nature14539). *Nature 2015*.

<h3 id="deep_sequence_learning">时空图神经网络（Spatial-Temporal Graph Neural Networks）</h3> 14-18

*	Feng J, Li Y, Zhang C, et al. 2018. [Deepmove: Predicting human mobility with attentional recurrent networks](https://dl.acm.org/doi/abs/10.1145/3178876.3186058) *Proceedings of the 2018 world wide web conference 2018*.
*	Gao J, Sharma R, Qian C, et al. 2021. [STAN: spatio-temporal attention network for pandemic prediction using real-world evidence](https://academic.oup.com/jamia/article-abstract/28/4/733/6118380). *Journal of the American Medical Informatics Association 2021*. 
*	Bai S, Kolter J Z, Koltun V. 2018. [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271). *arXiv: Learning 2018*.
*	Su Z, Liu T, Hao X, et al. 2023. [Spatial-temporal graph convolutional networks for traffic flow prediction considering multiple traffic parameters](https://link.springer.com/article/10.1007/s11227-023-05383-0). *The Journal of Supercomputing 2023*. 
*	Zhang K, Feng X, Wu L, et al. 2022. [Trajectory prediction for autonomous driving using spatial-temporal graph attention transformer](https://ieeexplore.ieee.org/abstract/document/9768029/). *IEEE Transactions on Intelligent Transportation Systems, 2022*.

<h3 id="probabilistic_deep_learning">概率深度学习</h3> 19-36

* Kingma D P, Welling M. 2013. [Auto-encoding variational bayes](https://arxiv.org/abs/1312.6114). *arXiv preprint 2013*.
* Ramchandran S, Tikhonov G, Lönnroth O, et al. 2024. [Learning conditional variational autoencoders with missing covariates](https://www.sciencedirect.com/science/article/pii/S0031320323008105). *Pattern Recognition 2024*.
*	HIGGINS I, MATTHEY L, PAL A, et al. 2017. [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://www.semanticscholar.org/paper/beta-VAE%3A-Learning-Basic-Visual-Concepts-with-a-Higgins-Matthey/a90226c41b79f8b06007609f39f82757073641e2). *International Conference on Learning Representations 2017*.
*	KIM H, MNIH A. 2018. [Disentangling by Factorising](https://proceedings.mlr.press/v80/kim18b.html?ref=https://githubhelp.com). *International Conference on Machine Learning 2018*.
*	Duan Y, Wang L, Zhang Q, et al. 2022. [Factorvae: A probabilistic dynamic factor model based on variational autoencoder for predicting cross-sectional stock returns](https://ojs.aaai.org/index.php/AAAI/article/view/20369). *Proceedings of the AAAI Conference on Artificial Intelligence 2022*.
* Sohl-Dickstein J, Weiss E, Maheswaranathan N, et al. 2015. [Deep unsupervised learning using nonequilibrium thermodynamics](https://proceedings.mlr.press/v37/sohl-dickstein15.html). *International conference on machine learning 2015*.
* Ho J, Jain A, Abbeel P. 2021. [Denoising diffusion probabilistic models](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html). *Advances in neural information processing systems 2020*.
* Song Y, Sohl-Dickstein J, Kingma D P, et al. 2020. [Score-based generative modeling through stochastic differential equations](https://arxiv.org/abs/2011.13456). *International Conference on Learning Representations 2021*. 
*	BAO Y, ZHOU H, HUANG S, et al. 2019. [Generating Sentences from Disentangled Syntactic and Semantic Spaces](https://arxiv.org/abs/1907.05789). *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019*.
* Lin L, Li Z, Li R, et al. 2023. [Diffusion models for time-series applications: a survey](https://link.springer.com/article/10.1631/FITEE.2300310). *Frontiers of Information Technology & Electronic Engineering 2023*.
* Lim H, Kim M, Park S, et al. 2023. [Regular time-series generation using sgm](https://arxiv.org/abs/2301.08518). *arXiv preprint 2023*. 
* Li Y, Lu X, Wang Y, et al. 2022. [Generative time series forecasting with diffusion, denoise, and disentanglement](https://proceedings.neurips.cc/paper_files/paper/2022/hash/91a85f3fb8f570e6be52b333b5ab017a-Abstract-Conference.html). *Advances in Neural Information Processing Systems 2022*. 
* Zhu Y, Yu J J, Zhao X, et al. 2024. [Controltraj: Controllable trajectory generation with topology-constrained diffusion model](https://arxiv.org/abs/2404.15380). *arXiv preprint 2024*. 
* Zhu Y, Ye Y, Zhang S, et al. 2024. [Difftraj: Generating gps trajectory with diffusion probabilistic model](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cd9b4a28fb9eebe0430c3312a4898a41-Abstract-Conference.html). Advances in *Neural Information Processing Systems 2024*.
* Wei T, Lin Y, Guo S, et al. 2024. [Diff-RNTraj: A Structure-aware Diffusion Model for Road Network-constrained Trajectory Generation](https://arxiv.org/abs/2402.07369). *arXiv preprint 2024*. 
* Larsen A B L, Sønderby S K, Larochelle H, et al. 2016. [Autoencoding beyond pixels using a learned similarity metric](https://proceedings.mlr.press/v48/larsen16). *International conference on machine learning 2016*. 
* Mescheder L, Nowozin S, Geiger A. 2017. [Adversarial variational bayes: Unifying variational autoencoders and generative adversarial networks](https://proceedings.mlr.press/v70/mescheder17a.html?ref=https://githubhelp.com). *International conference on machine learning. PMLR, 2017*.
* Makhzani A, Shlens J, Jaitly N, et al. 2015. [Adversarial autoencoders](https://arxiv.org/abs/1511.05644). *arXiv preprint arXiv:1511.05644, 2015*.

<h3 id="foundation_model_based_learning">基于底座模型的学习（Foundation Model-based Learning）</h3> 37-43

*	BROWN T B, MANN B, RYDER N, et al. 2020. [Language Models are Few-Shot Learners](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html). *arXiv: Computation and Language, 2020*.
*	Zhou T, Niu P, Sun L, et al. 2023. [One fits all: Power general time series analysis by pretrained lm](https://proceedings.neurips.cc/paper_files/paper/2023/hash/86c17de05579cde52025f9984e6e2ebb-Abstract-Conference.html). *Advances in neural information processing systems, 2023*.
* Wang P, Wei X, Hu F, et al. 2024. [TransGPT: Multi-modal Generative Pre-trained Transformer for Transportation](https://arxiv.org/abs/2402.07233). *arXiv preprint arXiv:2402.07233, 2024*.
*	ZHANG S, FU D, ZHANG Z, et al. 2024. [TrafficGPT: Viewing, Processing and Interacting with Traffic Foundation Models](https://www.sciencedirect.com/science/article/pii/S0967070X24000726). *Transport Policy, 2024*.
*	Liu R, Li C, Tang H, et al. 2024. [ST-LLM: Large Language Models Are Effective Temporal Learners](https://arxiv.org/abs/2404.00308). *arXiv preprint arXiv:2404.00308, 2024*.
* Xue H, Voutharoja B P, Salim F D. 2022. [Leveraging language foundation models for human mobility forecasting](https://dl.acm.org/doi/abs/10.1145/3557915.3561026). *Proceedings of the 30th International Conference on Advances in Geographic Information Systems 2022*.
* Yang C H H, Tsai Y Y, Chen P Y. 2021. [Voice2series: Reprogramming acoustic models for time series classification](https://proceedings.mlr.press/v139/yang21j.html). *International conference on machine learning. PMLR 2021*.

<h2 id="core_tasks">时空轨迹语义理解核心任务</h2> 

<h3 id="element_representation_learning">时空要素表示学习</h3> 44

* BENGIO Y, COURVILLE A, VINCENT P. 2013. [Representation Learning: A Review and New Perspectives](). *IEEE Transactions on Pattern Analysis and Machine Intelligence, 2013*.

<h4 id="deep_representation_learning">深度表示学习（Deep Representation Learning）</h3>  45-72

*	QUAN R, ZHU L, WU Y, et al. 2021. [Holistic LSTM for Pedestrian Trajectory Prediction](https://ieeexplore.ieee.org/abstract/document/9361440/). *IEEE Transactions on Image Processing, 2021*.
* LIN L, LI W, BI H, et al. 2022. [Vehicle Trajectory Prediction Using LSTMs With Spatial–Temporal Attention Mechanisms](https://ieeexplore.ieee.org/abstract/document/9349962/). *IEEE Intelligent Transportation Systems Magazine, 2022*.
*	KARIMZADEH M, AEBI R, SOUZA A M de, et al. 2021. [Reinforcement Learning-designed LSTM for Trajectory and Traffic Flow Prediction](https://ieeexplore.ieee.org/abstract/document/9417511). *2021 IEEE Wireless Communications and Networking Conference (WCNC). 2021*.
*	GUO H, RUI L lan, GAO Z peng. 2022. [V2V Task Offloading Algorithm with LSTM-based Spatiotemporal Trajectory Prediction Model in SVCNs](https://ieeexplore.ieee.org/abstract/document/9802720/). *IEEE Transactions on Vehicular Technology 2022*.
*	ZHANG C, NI Z, BERGER C. 2023. [Spatial-Temporal-Spectral LSTM: A Transferable Model for Pedestrian Trajectory Prediction](https://ieeexplore.ieee.org/abstract/document/10149368). *IEEE Transactions on Intelligent Vehicles, 2023*.
*	Zhou S, Li J, Wang H, et al. 2023. [GRLSTM: trajectory similarity computation with graph-based residual LSTM](https://ojs.aaai.org/index.php/AAAI/article/view/25624). *Proceedings of the AAAI Conference on Artificial Intelligence 2023*.
* Yang J, Chen Y, Du S, et al. 2024. [IA-LSTM: interaction-aware LSTM for pedestrian trajectory prediction](https://ieeexplore.ieee.org/abstract/document/10443050/). *IEEE transactions on cybernetics 2024*.
*	YANG C, PEI Z. 2023. [Long-Short Term Spatio-Temporal Aggregation for Trajectory Prediction](https://ieeexplore.ieee.org/abstract/document/10018105/). *IEEE Transactions on Intelligent Transportation Systems 2023*.
* OU J, JIN H, WANG X, et al. 2023. [STA-TCN: Spatial-temporal Attention over Temporal Convolutional Network for Next Point-of-interest Recommendation](https://dl.acm.org/doi/abs/10.1145/3596497). *ACM Transactions on Knowledge Discovery from Data 2023*.
*	Katariya V, Baharani M, Morris N, et al. 2022. [Deeptrack: Lightweight deep learning for vehicle trajectory prediction in highways](https://ieeexplore.ieee.org/abstract/document/9770480/). *IEEE Transactions on Intelligent Transportation Systems 2022*.
* Sadid H, Antoniou C. 2024. [Dynamic Spatio-temporal Graph Neural Network for Surrounding-aware Trajectory Prediction of Autonomous Vehicles](https://ieeexplore.ieee.org/abstract/document/10540254). *IEEE Transactions on Intelligent Vehicles, 2024*.
*	LV K, YUAN L. 2023. [SKGACN:Social Knowledge-guided Graph Attention Convolutional Network for Human Trajectory Prediction](https://ieeexplore.ieee.org/abstract/document/10145416/). *IEEE Transactions on Instrumentation and Measurement, 2023*.
* Yuan H, Zhang J, Zhang L, et al. 2023. [Vehicle Trajectory Prediction Based on Posterior Distributions Fitting and TCN-Transformer](https://ieeexplore.ieee.org/abstract/document/10366294/). *IEEE Transactions on Transportation Electrification, 2023*.
* JIANG R, XU H, GONG G, et al. 2022. [Spatial-Temporal Attentive LSTM for Vehicle-Trajectory Prediction](https://www.mdpi.com/2220-9964/11/7/354). *ISPRS International Journal of Geo-Information, 2022*.
*	HASAN F, HUANG H. 2023. [MALS-Net: A Multi-Head Attention-Based LSTM Sequence-to-Sequence Network for Socio-Temporal Interaction Modelling and Trajectory Prediction](https://www.mdpi.com/1424-8220/23/1/530). *Sensors 2023*.
*	LIANG Y, ZHAO Z. 2022. [NetTraj: A Network-Based Vehicle Trajectory Prediction Model With Directional Representation and Spatiotemporal Attention Mechanisms](https://ieeexplore.ieee.org/abstract/document/9629362/). *IEEE Transactions on Intelligent Transportation Systems, 2022*.
* ZHANG K, ZHAO L, DONG C, et al. 2023. [AI-TP: Attention-Based Interaction-Aware Trajectory Prediction for Autonomous Driving](https://ieeexplore.ieee.org/abstract/document/9723649/). *IEEE Transactions on Intelligent Vehicles, 2023*.
* Schmidt J, Jordan J, Gritschneder F, et al. 2022. [Crat-pred: Vehicle trajectory prediction with crystal graph convolutional neural networks and multi-head self-attention](https://ieeexplore.ieee.org/abstract/document/9811637/). *2022 International Conference on Robotics and Automation (ICRA). IEEE 2022*.
*	SCARSELLI F, GORI M,  AH CHUNG TSOI, et al. 2009. [The Graph Neural Network Model](https://ieeexplore.ieee.org/abstract/document/4700287/). *IEEE Transactions on Neural Networks 2009*.
*	ZHOU H, REN D, XIA H, et al. 2021. [AST-GNN: An attention-based spatio-temporal graph neural network for Interaction-aware pedestrian trajectory prediction](https://www.sciencedirect.com/science/article/pii/S092523122100388X). *Neurocomputing, 2021*.
*	ZHOU F, CHEN S, WU J, et al. 2021. [Trajectory-User Linking via Graph Neural Network](https://ieeexplore.ieee.org/abstract/document/9500836). *ICC 2021 - IEEE International Conference on Communications. 2021*. 
*	HAN P, WANG J, YAO D, et al. 2021. [A Graph-based Approach for Trajectory Similarity Computation in Spatial Networks](https://dl.acm.org/doi/abs/10.1145/3447548.3467337). *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &amp; Data Mining. 2021*.
* Zhu W, Liu Y, Wang P, et al. 2023. [Tri-HGNN: Learning triple policies fused hierarchical graph neural networks for pedestrian trajectory prediction](https://www.sciencedirect.com/science/article/pii/S0031320323004703). *Pattern Recognition, 2023*.
* Li R, Qin Y, Wang J, et al. 2023. [AMGB: Trajectory prediction using attention-based mechanism GCN-BiLSTM in IOV](https://www.sciencedirect.com/science/article/pii/S0167865523000715). *Pattern Recognition Letters, 2023*.
* Zhou F, Wang P, Xu X, et al. 2021. [Contrastive trajectory learning for tour recommendation](https://dl.acm.org/doi/full/10.1145/3462331). *ACM Transactions on Intelligent Systems and Technology (TIST), 2021*.
*	MAO Z, LI Z, LI D, et al. 2022. [Jointly Contrastive Representation Learning on Road Network and Trajectory](https://dl.acm.org/doi/abs/10.1145/3511808.3557370). *Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management. 2022*. 
* Deng L, Zhao Y, Fu Z, et al. 2022. [Efficient trajectory similarity computation with contrastive learning](https://dl.acm.org/doi/abs/10.1145/3511808.3557308). *Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 2022*.
* Chen Z, Zhang D, Feng S, et al. 2024. [KGTS: Contrastive Trajectory Similarity Learning over Prompt Knowledge Graph Embedding](https://ojs.aaai.org/index.php/AAAI/article/view/28672). *Proceedings of the AAAI Conference on Artificial Intelligence 2024*.

<h4 id="disentangled_representation_learning">解耦表示学习（Disentangled Representation Learning）</h3> 73-86

*	BENGIO Y, LAMBLIN P, POPOVICI D, et al. 2007. [Greedy Layer-Wise Training of Deep Networks](https://proceedings.neurips.cc/paper/2006/hash/5da713a690c067105aeb2fae32403405-Abstract.html). *Advances in Neural Information Processing Systems 19 2007*.
* Higgins I, Amos D, Pfau D, et al. 2018. [Towards a definition of disentangled representations](https://arxiv.org/abs/1812.02230). *arXiv preprint arXiv:1812.02230 2018*.
*	Goodfellow I, Pouget-Abadie J, Mirza M, et al. 2020. [Generative adversarial networks](https://dl.acm.org/doi/abs/10.1145/3422622). *Communications of the ACM 2020*.
* Shwartz-Ziv R, Tishby N. 2017. [Opening the black box of deep neural networks via information](https://arxiv.org/abs/1703.00810). *arXiv preprint arXiv:1703.00810 2017*.
*	ZHANG H, WU Y, TAN H, et al. 2022. [Understanding and Modeling Urban Mobility Dynamics via Disentangled Representation Learning](https://ieeexplore.ieee.org/abstract/document/9239884/). *IEEE Transactions on Intelligent Transportation Systems 2022*.
*	BAE I, JEON H G. 2022. [Disentangled Multi-Relational Graph Convolutional Network for Pedestrian Trajectory Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/16174). *Proceedings of the AAAI Conference on Artificial Intelligence 2022*.
* Fang Y, Qin Y, Luo H, et al. 2023. [STWave+: A Multi-Scale Efficient Spectral Graph Attention Network With Long-Term Trends for Disentangled Traffic Flow Forecasting](https://ieeexplore.ieee.org/abstract/document/10286992/). *IEEE Transactions on Knowledge and Data Engineering 2023*.
* Du Y, Guo X, Cao H, et al. 2022. [Disentangled spatiotemporal graph generative models](https://ojs.aaai.org/index.php/AAAI/article/view/20607). *Proceedings of the AAAI Conference on Artificial Intelligence 2022*.
* Fang Y, Qin Y, Luo H, et al. 2021. [Spatio-temporal meets wavelet: Disentangled traffic flow forecasting via efficient spectral graph attention network](https://arxiv.org/abs/2112.02740). *arXiv preprint arXiv:2112.02740 2021*.
* Wang Z, Zhu Y, Liu H, et al. 2022. [Learning graph-based disentangled representations for next POI recommendation](https://dl.acm.org/doi/abs/10.1145/3477495.3532012). *Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval 2022*.
* Qin Y, Wang Y, Sun F, et al. 2023. [DisenPOI: Disentangling sequential and geographical influence for point-of-interest recommendation](https://dl.acm.org/doi/abs/10.1145/3539597.3570408). *Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining 2023*.
* Gao Q, Hong J, Xu X, et al. 2023. [Predicting human mobility via self-supervised disentanglement learning](https://ieeexplore.ieee.org/abstract/document/10265198). *IEEE Transactions on Knowledge and Data Engineering 2023*.
*	QIN Y, GAO C, WANG Y, et al. 2022. [Disentangling Geographical Effect for Point-of-Interest Recommendation](https://ieeexplore.ieee.org/abstract/document/9947308/). *IEEE Transactions on Knowledge and Data Engineering 2022*.
* Tao H, Zeng J, Wang Z, et al. 2023. [Next POI Recommendation Based on Spatial and Temporal Disentanglement Representation](). *2023 IEEE International Conference on Web Services (ICWS). IEEE 2023*.

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


<h2 id="challenge_and_opportunities">新智能时代的时空轨迹语义理解的挑战与机遇</h2> 215

<h3 id="multimodal_data_processing">多模态数据处理</h3> 216-223

<h3 id="transparent_and_releasable">模型的透明可释</h3> 224-227
 
<h3 id="open_semantic_modeling">开放语义建模</h3> 228-233

<h3 id="availability_and_resource_consumption">模型可用与资源消耗</h3> 234-237

<h3 id="trust_issues">伦理与信任问题</h3> 238-240



