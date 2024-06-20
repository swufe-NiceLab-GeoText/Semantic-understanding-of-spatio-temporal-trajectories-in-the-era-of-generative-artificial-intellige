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

*	SWANSON N R, GRANGER C W J. 1997. [Impulse Response Functions Based on a Causal Approach to Residual Orthogonalization in Vector Autoregressions](https://www.tandfonline.com/doi/abs/10.1080/01621459.1997.10473634). *Journal of the American Statistical Association 1997*.
* HYVÄRINEN A, ZHANG K, SHIMIZU S, et al. 2010. [Estimation of a Structural Vector Autoregression Model Using Non-Gaussianity](https://www.jmlr.org/papers/volume11/hyvarinen10a/hyvarinen10a.pdf). *Journal of Machine Learning Research 2010*.
* Yao W, Sun Y, Ho A, et al. 2021. [Learning temporally causal latent processes from general temporal data](https://arxiv.org/abs/2110.05428). *arXiv preprint arXiv:2110.05428 2021*.
* Song X, Yao W, Fan Y, et al. 2024. [Temporally Disentangled Representation Learning under Unknown Nonstationarity](https://proceedings.neurips.cc/paper_files/paper/2023/hash/19a567abaec3990cb40d7a013556fecd-Abstract-Conference.html). *Advances in Neural Information Processing Systems, 2024*.
*	HUANG B, ZHANG K, ZHANG J, et al. 2019. [Causal Discovery from Heterogeneous/Nonstationary Data with Independent Changes](https://www.jmlr.org/papers/v21/19-232.html). *arXiv: Learning, 2019*.
*	ZHANG K, GONG M, STOJANOV P, et al. 2020. [Domain Adaptation as a Problem of Inference on Graphical Models](https://proceedings.neurips.cc/paper/2020/hash/3430095c577593aad3c39c701712bcfe-Abstract.html). *Neural Information Processing Systems, 2020*.

<h3 id="trajectory_similarity_learning">基于时空语义理解的轨迹相似学习</h2> 

<h4 id="trajectory_clustering">时空轨迹聚类</h4> 93-106

*	KRISHNA K, NARASIMHA MURTY M. 1999. [Genetic K-means algorithm](https://ieeexplore.ieee.org/abstract/document/764879/). *IEEE Transactions on Systems, Man and Cybernetics, Part B (Cybernetics), 1999*. 
*	ESTER M, KRIEGEL H P, SANDER J, et al. 1996. [A density-based algorithm for discovering clusters in large spatial Databases with Noise](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf?source=post_page---------------------------). *Knowledge Discovery and Data Mining, 1996*.
*	Ankerst M, Breunig M M, Kriegel H P, et al. 1999. [OPTICS: Ordering points to identify the clustering structure](https://dl.acm.org/doi/abs/10.1145/304181.304187). *ACM Sigmod record, 1999*.
* Zhang T, Ramakrishnan R, Livny M. 1996. [BIRCH: an efficient data clustering method for very large databases](https://dl.acm.org/doi/abs/10.1145/235968.233324). *ACM sigmod record, 1996*.
* Wang W, Yang J, Muntz R. 1997. [STING: A statistical information grid approach to spatial data mining](http://cs.bme.hu/~marti/adatbanya/STING.pdf). *Vldb. 1997*.
*	BIRANT D, KUT A. 2007. [ST-DBSCAN: An algorithm for clustering spatial–temporal data](https://www.sciencedirect.com/science/article/pii/S0169023X06000218). *Data &amp; Knowledge Engineering, 2007*. 
* Malhan A, Gunturi V M V, Naik V. 2017. [ST-OPTICS: A spatial-temporal clustering algorithm with time recommendations for taxi services](https://repository.iiitd.edu.in/jspui/handle/123456789/529). *2017*.
*	LI X, ZHAO K, CONG G, et al. 2018. [Deep Representation Learning for Trajectory Similarity Computation](https://ieeexplore.ieee.org/abstract/document/8509283/). *2018 IEEE 34th International Conference on Data Engineering (ICDE). 2018*.
*	YUE M, LI Y, YANG H, et al. 2020. [DETECT: Deep Trajectory Clustering for Mobility-Behavior Analysis](https://ieeexplore.ieee.org/abstract/document/9006561/). *Cornell University - arXiv, 2020*.
*	FANG Z, DU Y, CHEN L, et al. 2021. [E2DTC: An End to End Deep Trajectory Clustering Framework via Self-Training](https://ieeexplore.ieee.org/abstract/document/9458936). *2021 IEEE 37th International Conference on Data Engineering (ICDE). 2021*.
* Si J, Yang J, Xiang Y, et al. 2024. [ConDTC: Contrastive Deep Trajectory Clustering for Fine-grained Mobility Pattern Mining](https://ieeexplore.ieee.org/abstract/document/10433249). *IEEE Transactions on Big Data, 2024*.
*	JIANG Q, LIU Y, DING Z, et al. 2023. [Behavior pattern mining based on spatiotemporal trajectory multidimensional information fusion](https://www.sciencedirect.com/science/article/pii/S1000936122002503). *Chinese Journal of Aeronautics, 2023*.
*	WANG W, XIA F, NIE H, et al. 2021. [Vehicle Trajectory Clustering Based on Dynamic Representation Learning of Internet of Vehicles](https://ieeexplore.ieee.org/abstract/document/9115819/). *IEEE Transactions on Intelligent Transportation Systems, 2021*.
*	HOSEINI F, RAHROVANI S, CHEHREGHANI M H. 2021. [Vehicle Motion Trajectories Clustering via Embedding Transitive Relations](). *2021 IEEE International Intelligent Transportation Systems Conference (ITSC). 2021*. 

<h4 id="trajectory_classification">时空轨迹分类</h4> 107-121

*	OH J, LIM K T, CHUNG Y S. 2021. [TrajNet: An Efficient and Effective Neural Network for Vehicle Trajectory Classification](https://www.scitepress.org/Papers/2021/102433/102433.pdf). *Proceedings of the 10th International Conference on Pattern Recognition Applications and Methods 2021*.
*	FREITAS N C A de, DA SILVA T L C, DE MACÊDO J A F, et al. 2021. [Using deep learning for trajectory classification in imbalanced dataset](https://journals.flvc.org/FLAIRS/article/view/128368). *The International FLAIRS Conference Proceedings 2021*.
* GUO T, XIE L. 2022. [Research on Ship Trajectory Classification Based on a Deep Convolutional Neural Network](https://www.mdpi.com/2077-1312/10/5/568). *Journal of Marine Science and Engineering 2022*.
* BAE K, LEE S, LEE W. 2022. [Transformer Networks for Trajectory Classification](https://ieeexplore.ieee.org/abstract/document/9736500/). *2022 IEEE International Conference on Big Data and Smart Computing (BigComp) 2022*.
* Liang Y, Ouyang K, Wang Y, et al. 2022. [TrajFormer: Efficient trajectory classification with transformers](https://dl.acm.org/doi/abs/10.1145/3511808.3557481). *Proceedings of the 31st ACM International Conference on Information & Knowledge Management 2022*.
*	JIN C, TAO T, LUO X, et al. 2020. [S2N2: An Interpretive Semantic Structure Attention Neural Network for Trajectory Classification](https://ieeexplore.ieee.org/abstract/document/9044862/). *IEEE Access, 2020*.
*	FERRERO C A, PETRY L M, ALVARES L O, et al. 2020. [MasterMovelets: discovering heterogeneous movelets for multiple aspect trajectory classification](https://link.springer.com/article/10.1007/s10618-020-00676-x). *Data Mining and Knowledge Discovery, 2020*.
*	MAKRIS A, KONTOPOULOS I, PSOMAKELIS E, et al. 2021. [Semi-supervised trajectory classification using convolutional auto-encoders](https://dl.acm.org/doi/abs/10.1145/3486637.3489492). *Proceedings of the 1st ACM SIGSPATIAL International Workshop on Animal Movement Ecology and Human Mobility. 2021*.
*	AHMED U, SRIVASTAVA G, DJENOURI Y, et al. 2021. [Knowledge graph based trajectory outlier detection in sustainable smart cities](https://www.sciencedirect.com/science/article/pii/S2210670721008453). *Sustainable Cities and Society, 2022*.
*	LANDI C, SPINNATO F, GUIDOTTI R, et al.2023. [Geolet: An Interpretable Model for Trajectory Classification](https://link.springer.com/chapter/10.1007/978-3-031-30047-9_19). *Advances in Intelligent Data Analysis XXI,Lecture Notes in Computer Science. 2023*.
*	HU X, HAN Y, GENG Z. 2021. [Novel Trajectory Representation Learning Method and Its Application to Trajectory-User Linking](https://ieeexplore.ieee.org/abstract/document/9478304/). *IEEE Transactions on Instrumentation and Measurement, 2021*.
* Chen W, Li S, Huang C, et al. 2022. [Mutual distillation learning network for trajectory-user linking](https://arxiv.org/abs/2205.03773). *arXiv preprint 2022*.
* Alsaeed M, Agrawal A, Papagelis M. 2023. [Trajectory-User Linking using Higher-order Mobility Flow Representations](https://ieeexplore.ieee.org/abstract/document/10214932/). *2023 24th IEEE International Conference on Mobile Data Management (MDM). IEEE 2023*.
* Deng L, Sun H, Zhao Y, et al. 2023. [S2tul: A semi-supervised framework for trajectory-user linking](https://dl.acm.org/doi/abs/10.1145/3539597.3570410). *Proceedings of the sixteenth ACM international conference on web search and data mining 2023*.
* Chen W, Huang C, Yu Y, et al. 2024. [Trajectory-User Linking via Hierarchical Spatio-Temporal Attention Networks](https://dl.acm.org/doi/abs/10.1145/3635718). *ACM Transactions on Knowledge Discovery from Data 2024*.

<h4 id="abnormal_identification">异常行为识别</h4> 122-133

* KUMARAN SANTHOSH K, DOGRA D P, ROY P P, et al. 2022. [Vehicular Trajectory Classification and Traffic Anomaly Detection in Videos Using a Hybrid CNN-VAE Architecture](https://ieeexplore.ieee.org/abstract/document/9531567/). *IEEE Transactions on Intelligent Transportation Systems 2022*.
* Wiederer J, Bouazizi A, Troina M, et al. 2022. [Anomaly detection in multi-agent trajectories for automated driving](https://proceedings.mlr.press/v164/wiederer22a.html). *Conference on Robot Learning. PMLR 2022*.
*	SU Y, YAO D, TIAN T, et al. 2023. [Transfer learning for region-wide trajectory outlier detection](https://ieeexplore.ieee.org/abstract/document/10179898/). *IEEE Access 2023*. 
*	XIE L, GUO T, CHANG J, et al. 2023. [A Novel Model for Ship Trajectory Anomaly Detection Based on Gaussian Mixture Variational Autoencoder](https://ieeexplore.ieee.org/abstract/document/10151936/). *IEEE Transactions on Vehicular Technology 2023*.
* Han X, Cheng R, Ma C, et al. 2022. [DeepTEA: Effective and efficient online time-dependent trajectory outlier detection](https://dl.acm.org/doi/abs/10.14778/3523210.3523225). *Proceedings of the VLDB Endowment 2022*.
* Singh S K, Fowdur J S, Gawlikowski J, et al. 2021. [Leveraging Graph and Deep Learning Uncertainties to Detect Anomalous Trajectories](https://arxiv.org/abs/2107.01557). *arXiv preprint 2021*.
*	SHI Y, WANG D, NI Z, et al. 2022. [A Sequential Pattern Mining Based Approach to Adaptively Detect Anomalous Paths in Floating Vehicle Trajectories](https://ieeexplore.ieee.org/abstract/document/9762798/). *IEEE Transactions on Intelligent Transportation Systems 2022*.
* Djenouri Y, Djenouri D, Lin J C W. 2021. [Trajectory outlier detection: New problems and solutions for smart cities](https://dl.acm.org/doi/abs/10.1145/3425867). *ACM Transactions on Knowledge Discovery from Data (TKDD) 2021*.
*	MAO J, LIU J, JIN C, et al. 2021. [Feature Grouping–based Trajectory Outlier Detection over Distributed Streams](https://dl.acm.org/doi/abs/10.1145/3444753). *ACM Transactions on Intelligent Systems and Technology 2021*.
*	AHMED U, SRIVASTAVA G, DJENOURI Y, et al. 2022. [Deviation Point Curriculum Learning for Trajectory Outlier Detection in Cooperative Intelligent Transport Systems](https://ieeexplore.ieee.org/abstract/document/9646484/). *IEEE Transactions on Intelligent Transportation Systems, 2022*.
* Li C, Feng G, Jia Y, et al. 2023. [RETAD: Vehicle Trajectory Anomaly Detection Based on Reconstruction Error](https://www.igi-global.com/pdf.aspx?tid=316460&ptid=315821&ctid=4&oa=true&isxn=9781668488157). *International Journal of Data Warehousing & Mining 2023*.
*	GAO L, XU C, WANG F, et al. 2023. [Flight data outlier detection by constrained LSTM-autoencoder](). *Wireless Networks, 2023*.

<h3 id="trajectory_prediction_learning">基于时空语义理解的轨迹预测学习</h3> 

<h4 id="space_time_trajectory_prediction">时空轨迹预测</h4> 134-174

*	WANG H, SHEN H, OUYANG W, et al. 2018. [Exploiting POI-Specific Geographical Influence for Point-of-Interest Recommendation](https://www.ijcai.org/proceedings/2018/0539.pdf). *Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence 2018*.
*	FENG S, CONG G, AN B, et al. 2022. [POI2Vec: Geographical Latent Representation for Predicting Future Visitors](https://ojs.aaai.org/index.php/AAAI/article/view/10500). *Proceedings of the AAAI Conference on Artificial Intelligence 2022*.
*	RENDLE S, FREUDENTHALER C, SCHMIDT-THIEME L. 2010. [Factorizing personalized Markov chains for next-basket recommendation](https://dl.acm.org/doi/abs/10.1145/1772690.1772773). *Proceedings of the 19th international conference on World wide web 2010*.
*	ZHAO K, ZHANG Y, YIN H, et al. 2020. [Discovering Subsequence Patterns for Next POI Recommendation](https://www.ijcai.org/Proceedings/2020/0445.pdf). *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence 2020*.
* TANG Q, YANG M, YANG Y. 2019. [ST-LSTM: A Deep Learning Approach Combined Spatio-Temporal Features for Short-Term Forecast in Rail Transit](https://onlinelibrary.wiley.com/doi/full/10.1155/2019/8392592). *Journal of Advanced Transportation 2019*.
* Dai S, Yu Y, Fan H, et al. 2022. [Spatio-temporal representation learning with social tie for personalized POI recommendation](https://link.springer.com/article/10.1007/s41019-022-00180-w). *Data Science and Engineering 2022*.
*	LI M, ZHENG W, XIAO Y, et al. 2020. [An Adaptive POI Recommendation Algorithm by Integrating User’s Temporal and Spatial Features in LBSNs](https://dl.acm.org/doi/abs/10.1145/3414274.3414494). *Proceedings of the 3rd International Conference on Data Science and Information Technology 2020*.
*	TAHMASBI H, JALALI M, SHAKERI H. 2021. [Modeling user preference dynamics with coupled tensor factorization for social media recommendation](https://link.springer.com/article/10.1007/s12652-020-02714-4). *Journal of Ambient Intelligence and Humanized Computing 2021*.
* Wang X, Sun G, Fang X, et al. 2022. [Modeling Spatio-temporal Neighbourhood for Personalized Point-of-interest Recommendation](https://www.ijcai.org/proceedings/2022/0490.pdf). *IJCAI. 2022*.
*	PARVEEN R, VARMA N S. 2021. [Friend’s recommendation on social media using different algorithms of machine learning](https://www.sciencedirect.com/science/article/pii/S2666285X21000406). *Global Transitions Proceedings 2021*. 
*	SARASWATHI K, MOHANRAJ V, SURESH Y, et al. 2023. [Deep Learning Enabled Social Media Recommendation Based on User Comments](https://cdn.techscience.cn/ueditor/files/csse/TSP_CSSE-44-2/TSP_CSSE_27987/TSP_CSSE_27987.pdf). *Computer Systems Science and Engineering 2023*.
*	LIU H, TONG Y, HAN J, et al. 2022. [Incorporating Multi-Source Urban Data for Personalized and Context-Aware Multi-Modal Transportation Recommendation](https://ieeexplore.ieee.org/abstract/document/9063461/). *IEEE Transactions on Knowledge and Data Engineering 2022*.
*	ZHU M, HU J, HAO H, et al. 2019. [Personalized Context-Aware Multi-Modal Transportation Recommendation](https://arxiv.org/abs/1910.12601). *arXiv: Computers and Society 2019*.
*	LIU Y, LI K, YAN D, et al. 2023. [The prediction of disaster risk paths based on IECNN model](https://link.springer.com/article/10.1007/s11069-023-05855-9). *Natural Hazards 2023*. 
*	ELTEHEWY R, ABOUELFARAG A, SALEH S N. 2023. [Efficient Classification of Imbalanced Natural Disasters Data Using Generative Adversarial Networks for Data Augmentation](https://www.mdpi.com/2220-9964/12/6/245). *ISPRS International Journal of Geo-Information 2023*.
* Zeng C, Bertsimas D. 2023. [Global flood prediction: a multimodal machine learning approach](https://arxiv.org/abs/2301.12548). *arXiv preprint arXiv:2301.12548 2023*.
*	DIKSHIT A, PRADHAN B, ALAMRI A M. 2021. [Long lead time drought forecasting using lagged climate variables and a stacked long short-term memory model](https://www.sciencedirect.com/science/article/pii/S0048969720361672). *Science of The Total Environment 2021*.
*	MOKHTAR A, JALALI M, HE H, et al. 2021. [Estimation of SPEI Meteorological Drought Using Machine Learning Algorithms](https://ieeexplore.ieee.org/abstract/document/9408611/). *IEEE Access 2021*.
*	DANANDEH MEHR A, RIKHTEHGAR GHIASI A, YASEEN Z M, et al. 2023. [A novel intelligent deep learning predictive model for meteorological drought forecasting](https://link.springer.com/article/10.1007/s12652-022-03701-7). *Journal of Ambient Intelligence and Humanized Computing, 2023*.
* Bi K, Xie L, Zhang H, et al. 2022. [Pangu-weather: A 3d high-resolution model for fast and accurate global weather forecast](https://arxiv.org/abs/2211.02556). *arXiv preprint 2022*.
* Bi K, Xie L, Zhang H, et al. 2023. [Accurate medium-range global weather forecasting with 3D neural networks](https://www.nature.com/articles/s41586-023-06185-3). *Nature 2023*.
* Xu Z, Wei X, Hao J, et al. 2024. [DGFormer: a physics-guided station level weather forecasting model with dynamic spatial-temporal graph neural network](https://link.springer.com/article/10.1007/s10707-024-00511-1). *GeoInformatica 2024*.
* Lam R, Sanchez-Gonzalez A, Willson M, et al. 2022. [GraphCast: Learning skillful medium-range global weather forecasting](https://arxiv.org/abs/2212.12794). *arXiv preprint 2022*.
* Pathak J, Subramanian S, Harrington P, et al. 2022. [Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators](https://arxiv.org/abs/2202.11214). *arXiv preprint 2022*.
* Kurth T, Subramanian S, Harrington P, et al. 2023. [Fourcastnet: Accelerating global high-resolution weather forecasting using adaptive fourier neural operators](https://dl.acm.org/doi/abs/10.1145/3592979.3593412). *Proceedings of the platform for advanced scientific computing conference 2023*.
*	WEYN J A, DURRAN D R, CARUANA R, et al. 2021. [Sub‐Seasonal Forecasting With a Large Ensemble of Deep‐Learning Weather Prediction Models](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002502). *Journal of Advances in Modeling Earth Systems 2021*.
*	NI Q, WANG Y, FANG Y. 2022. [GE-STDGN: a novel spatio-temporal weather prediction model based on graph evolution](https://link.springer.com/article/10.1007/s10489-021-02824-2). *Applied Intelligence 2022*. 
* Hu Y, Chen L, Wang Z, et al. 2023. [SwinVRNN: A Data‐Driven Ensemble Forecasting Model via Learned Distribution Perturbation](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003211). *Journal of Advances in Modeling Earth Systems 2023*.
* Nguyen T, Brandstetter J, Kapoor A, et al. 2023. [Climax: A foundation model for weather and climate](https://arxiv.org/abs/2301.10343). *arXiv preprint 2023*.
* Chen K, Han T, Gong J, et al. 2023. [Fengwu: Pushing the skillful global medium-range weather forecast beyond 10 days lead](https://arxiv.org/abs/2304.02948). *arXiv preprint 2023*.
*	DANG W, WANG H, PAN S, et al. 2022. [Predicting Human Mobility via Graph Convolutional Dual-attentive Networks](). *Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining 2022*.
*	FRITZ C, DORIGATTI E, RÜGAMER D. 2022. [Combining graph neural networks and spatio-temporal disease models to improve the prediction of weekly COVID-19 cases in Germany](https://www.nature.com/articles/s41598-022-07757-5). *Scientific Reports 2022*. 
* Niraula P, Mateu J, Chaudhuri S. 2022. [A Bayesian machine learning approach for spatio-temporal prediction of COVID-19 cases](https://link.springer.com/article/10.1007/s00477-021-02168-w). *Stochastic Environmental Research and Risk Assessment 2022*.
* Liu Y, Rong Y, Guo Z, et al. 2023. [Human mobility modeling during the COVID-19 pandemic via deep graph diffusion infomax](https://ojs.aaai.org/index.php/AAAI/article/view/26678). *Proceedings of the AAAI Conference on Artificial Intelligence 2023*.
* Ma Y, Gerard P, Tian Y, et al. 2022. [Hierarchical spatio-temporal graph neural networks for pandemic forecasting](https://dl.acm.org/doi/abs/10.1145/3511808.3557350). *Proceedings of the 31st ACM International Conference on Information & Knowledge Management 2022*.
* Li Y, Fan Z, Zhang J, et al. 2022. [Heterogeneous hypergraph neural network for friend recommendation with human mobility](https://dl.acm.org/doi/abs/10.1145/3511808.3557609). *Proceedings of the 31st ACM International Conference on Information & Knowledge Management 2022*.
* PAN X, CAI X, SONG K, et al. 2023. [Location Recommendation Based on Mobility Graph With Individual and Group Influences](http://dx.doi.org/10.1109/tits.2022.3149869). *IEEE Transactions on Intelligent Transportation Systems 2023*. 
*	TERROSO-SÁENZ F, MUÑOZ A. 2022. [Nation-wide human mobility prediction based on graph neural networks](http://dx.doi.org/10.1007/s10489-021-02645-3). *Applied Intelligence, 2022*. 
* Kong X, Wang K, Hou M, et al. 2022. [Exploring human mobility for multi-pattern passenger prediction: A graph learning framework](https://ieeexplore.ieee.org/abstract/document/9709191/). *IEEE Transactions on Intelligent Transportation Systems 2022*.
* Solatorio A V. 2023. [GeoFormer: Predicting Human Mobility using Generative Pre-trained Transformer (GPT)](https://dl.acm.org/doi/abs/10.1145/3615894.3628499). *Proceedings of the 1st International Workshop on the Human Mobility Prediction Challenge. 2023*.
*	CHOYA T, TAMURA N, KATAYAMA S, et al. 2023. [CrowdFlowTransformer: Capturing Spatio-Temporal Dependence for Forecasting Human Mobility](https://ieeexplore.ieee.org/abstract/document/10150301/). *2023 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops) 2023*. 


<h4 id="space_time_event_prediction">时空事件预测</h4> 175-202

*	XU L, CHEN N, CHEN Z, et al. 2021. [Spatiotemporal forecasting in earth system science: Methods, uncertainties, predictability and future directions](https://www.sciencedirect.com/science/article/pii/S0012825221003299). *Earth-Science Reviews 2021*.
* Yang G, Yu H, Xi H. 2022. [A Spatio-Temporal Traffic Flow Prediction Method Based on Dynamic Graph Convolution Network](https://ieeexplore.ieee.org/abstract/document/10033842). *2022 34th Chinese Control and Decision Conference (CCDC). IEEE, 2022*.
* Cheng M, Jiang G P, Song Y, et al. 2022. [Dynamic Spatio-temporal traffic flow prediction based on multi fusion graph attention network](https://ieeexplore.ieee.org/abstract/document/9902776). *2022 41st Chinese Control Conference (CCC). IEEE, 2022*.
* Zhou B, Zhou H, Wang W, et al. 2024. [HDM-GNN: A Heterogeneous Dynamic Multi-view Graph Neural Network for Crime Prediction](https://dl.acm.org/doi/abs/10.1145/3665141). *ACM Transactions on Sensor Networks 2024*.
* Sun Y, Chen T, Yin H. 2023. [Spatial-temporal meta-path guided explainable crime prediction](https://link.springer.com/article/10.1007/s11280-023-01137-3). *World Wide Web, 2023*.
* Liang W, Cao J, Chen L, et al. 2023. [Crime prediction with missing data via spatiotemporal regularized tensor decomposition](https://ieeexplore.ieee.org/abstract/document/10145042/). *IEEE Transactions on Big Data 2023*.
* Wang C, Lin Z, Yang X, et al. 2022. [Hagen: Homophily-aware graph convolutional recurrent network for crime forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/20338). *Proceedings of the AAAI Conference on Artificial Intelligence 2022*.
* Yang C. 2023. [TransCrimeNet: A Transformer-Based Model for Text-Based Crime Prediction in Criminal Networks](https://arxiv.org/abs/2311.09529). *arXiv preprint 2023*.
*	HOU M, HU X, CAI J, et al. 2022. [An Integrated Graph Model for Spatial–Temporal Urban Crime Prediction Based on Attention Mechanism](https://www.mdpi.com/2220-9964/11/5/294). *ISPRS International Journal of Geo-Information 2022*.
*	GANDAPUR M Q. 2022. [E2E-VSDL: End-to-end video surveillance-based deep learning model to detect and prevent criminal activities](https://www.sciencedirect.com/science/article/pii/S0262885622000968). *Image and Vision Computing 2022*.
* RAYHAN Y, HASHEM T. 2023. [AIST: An Interpretable Attention-Based Deep Learning Model for Crime Prediction](https://dl.acm.org/doi/abs/10.1145/3582274). *ACM Transactions on Spatial Algorithms and Systems 2023*.
* Li Z, Huang C, Xia L, et al. 2022. [Spatial-temporal hypergraph self-supervised learning for crime prediction](https://ieeexplore.ieee.org/abstract/document/9835423/). *2022 IEEE 38th International Conference on Data Engineering (ICDE). IEEE 2022*.
*	XIA L, HUANG C, XU Y, et al. 2021. [Spatial-Temporal Sequential Hypergraph Network for Crime Prediction with Dynamic Multiplex Relation Learning](https://arxiv.org/abs/2201.02435). *Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence. 2021*.
[188]	GAO Q, FU H W, WEI Y, et al. Spatial-Temporal Diffusion Probabilistic Learning for Crime Prediction[J]. 2023.
* Amato F, Guignard F, Robert S, et al. 2020. [A novel framework for spatio-temporal prediction of environmental data using deep learning](https://www.nature.com/articles/s41598-020-79148-7). *Scientific reports, 2020*.
* Li T, Zhang J, Bao K, et al. 2020. [Autost: Efficient neural architecture search for spatio-temporal prediction](https://dl.acm.org/doi/abs/10.1145/3394486.3403122). *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020*.
*	WANG Y, TANG S, LEI Y, et al. 2020. [DisenHAN: Disentangled Heterogeneous Graph Attention Network for Recommendation](https://dl.acm.org/doi/abs/10.1145/3340531.3411996). *Proceedings of the 29th ACM International Conference on Information &amp; Knowledge Management 2020*.
*	HUANG Z, MA J, DONG Y, et al. 2022. [Empowering Next POI Recommendation with Multi-Relational Modeling](https://dl.acm.org/doi/abs/10.1145/3477495.3531801). *Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval 2022*.
* LIU X, YANG Y, XU Y, et al. 2022. [Real-time POI recommendation via modeling long- and short-term user preferences](https://www.sciencedirect.com/science/article/pii/S092523122101434X). *Neurocomputing 2022*. 
*	WANG Z, ZHU Y, ZHANG Q, et al. 2022. [Graph-Enhanced Spatial-Temporal Network for Next POI Recommendation](https://dl.acm.org/doi/abs/10.1145/3513092). *ACM Transactions on Knowledge Discovery from Data 2022*.
[195]	LI Q, XU X, LIU X, et al. An Attention-Based Spatiotemporal GGNN for Next POI Recommendation[J].
[196]	WANG X, FUKUMOTO F, LI J, et al. STaTRL: Spatial-temporal and text representation learning for POI recommendation[J].
[197]	WANG K, WANG X, LU X. POI recommendation method using LSTM-attention in LBSN considering privacy protection[J/OL]. Complex &amp; Intelligent Systems, 2023: 2801-2812. http://dx.doi.org/10.1007/s40747-021-00440-8. DOI:10.1007/s40747-021-00440-8.
[198]	Spatio-Temporal Hypergraph Learning for Next POI Recommendation[J].
[199]	WANG X, FUKUMOTO F, CUI J, et al. EEDN: Enhanced Encoder-Decoder Network with Local and Global Context Learning for POI Recommendation[J].
[200]	FU J, GAO R, YU Y, et al. Contrastive graph learning long and short-term interests for POI recommendation[J].
[201]	LANG C, WANG Z, HE K, et al. POI recommendation based on a multiple bipartite graph network model[J/OL]. The Journal of Supercomputing, 2022, 78(7): 9782-9816. http://dx.doi.org/10.1007/s11227-021-04279-1. DOI:10.1007/s11227-021-04279-1.
[202]	QIN Y, WU H, JU W, et al. A Diffusion model for POI recommendation[J]. 2023.


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


* 城感1
* 以下为当前文中43、44、46、47



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



