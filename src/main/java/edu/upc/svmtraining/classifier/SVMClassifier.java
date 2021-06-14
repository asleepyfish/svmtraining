package edu.upc.svmtraining.classifier;

import com.hankcs.hanlp.classification.corpus.Document;
import com.hankcs.hanlp.classification.corpus.IDataSet;
import com.hankcs.hanlp.classification.corpus.MemoryDataSet;
import com.hankcs.hanlp.classification.features.BaseFeatureData;
import com.hankcs.hanlp.classification.features.ChiSquareFeatureExtractor;
import com.hankcs.hanlp.classification.features.DfFeatureData;
import com.hankcs.hanlp.classification.models.AbstractModel;
import com.hankcs.hanlp.classification.tokenizers.ITokenizer;
import com.hankcs.hanlp.classification.utilities.CollectionUtility;
import com.hankcs.hanlp.classification.utilities.MathUtility;
import com.hankcs.hanlp.collection.trie.ITrie;
import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import de.bwaldvogel.liblinear.*;
import edu.upc.svmtraining.model.SVMModel;
import edu.upc.svmtraining.util.TfIdfFeatureWeighterUtil;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import static com.hankcs.hanlp.classification.utilities.Predefine.logger;


public class SVMClassifier {
    public SVMModel model;

    public final Map<Integer, String> WORD_MAP = new HashMap<>();

    public SVMClassifier() {
    }

    public SVMClassifier(SVMModel model) {
        this.model = model;
    }

    public SVMModel getModel() {
        return model;
    }


    /**
     * 提取指定路径指定编码的训练集到数据集对象,然后进行训练
     *
     * @param folderPath  训练集路径
     * @param charsetName 训练集编码
     * @throws IOException Exception
     */
    public void train(String folderPath, String charsetName) throws IOException {
        IDataSet dataSet = new MemoryDataSet();
        dataSet.load(folderPath, charsetName);//从给定文件夹路径加载文件
        // 选择特征
        DfFeatureData featureData = selectFeatures(dataSet);
        // 构造权重计算逻辑
        TfIdfFeatureWeighterUtil weighter = new TfIdfFeatureWeighterUtil(dataSet.size(), featureData.df);
        // 构造SVM问题
        Problem problem = createLiblinearProblem(dataSet, featureData, weighter);
        // 释放内存
        BinTrie<Integer> wordIdTrie = featureData.wordIdTrie;
        ITokenizer tokenizer = dataSet.getTokenizer();
        String[] catalog = dataSet.getCatalog().toArray();
        System.gc();
        // 求解SVM问题
        Model svmModel = solveLibLinearProblem(problem);
        // 保留训练过程中数据
        model = new SVMModel();
        model.tokenizer = tokenizer;
        model.wordIdTrie = wordIdTrie;
        model.catalog = catalog;
        model.svmModel = svmModel;
        model.featureWeighter = weighter;
    }

    /**
     * 统计特征并且执行特征选择，返回一个featureData对象，用于计算模型中的概率
     *
     * @param dataSet 训练集对象
     * @return featureData对象
     */
    public DfFeatureData selectFeatures(IDataSet dataSet) {
        ChiSquareFeatureExtractor csfe = new ChiSquareFeatureExtractor();

        //featureData对象包含文档中所有特征及其统计信息
        DfFeatureData featureData = new DfFeatureData(dataSet); //执行统计

        logger.start("使用卡方检测选择特征中...");
        //我们传入这些统计信息到特征选择算法中,得到选择的特征与其权重
        Map<Integer, Double> selectedFeatures = csfe.chi_square(featureData);

        //从训练数据中删掉无用的特征并重建特征映射表
        String[] wordIdArray = dataSet.getLexicon().getWordIdArray();//得到特征词的标志id字符串数组
        int[] idMap = new int[wordIdArray.length];
        Arrays.fill(idMap, -1);
        featureData.wordIdTrie = new BinTrie<>();
        featureData.df = new int[selectedFeatures.size()];//df自由度指的是计算某一统计量时，取值不受限制的变量个数
        int p = -1;
        for (Integer feature : selectedFeatures.keySet()) {
            ++p;
            featureData.wordIdTrie.put(wordIdArray[feature], p);
            featureData.df[p] = MathUtility.sum(featureData.featureCategoryJointCount[feature]);
            idMap[feature] = p;
        }
        logger.finish(",选中特征数:%d / %d = %.2f%%\n", selectedFeatures.size(), featureData.featureCategoryJointCount.length, MathUtility.percentage(selectedFeatures.size(), featureData.featureCategoryJointCount.length));
        dataSet.shrink(idMap);//将idMap中value不为-1的加入tfMap中
        return featureData;
    }

    /**
     * 使用liblinear创建SVM问题
     *
     * @param dataSet         训练集对象
     * @param baseFeatureData 选择特征对象
     * @param weighter        权重计算对象
     * @return 返回svm问题对象
     */
    public Problem createLiblinearProblem(IDataSet dataSet, BaseFeatureData baseFeatureData, TfIdfFeatureWeighterUtil weighter) {
        Problem problem = new Problem();
        int n = dataSet.size();
        problem.l = n;//训练样本数
        problem.n = baseFeatureData.featureCategoryJointCount.length;//特征维数(特征数量)
        problem.x = new FeatureNode[n][];//特征数据
        problem.y = new double[n];  //类别, liblinear的y数组是浮点数
        Iterator<Document> iterator = dataSet.iterator();
        for (int i = 0; i < n; i++) {
            // 构造文档向量
            Document document = iterator.next();
            problem.x[i] = buildDocumentVector(document, weighter);
            // 设置样本的y值
            problem.y[i] = document.category;
        }

        return problem;
    }

    /**
     * 构建文档向量,处理特征数据,计算特征权重
     *
     * @param document 文档对象
     * @param weighter 权重计算对象
     * @return 特征节点数组
     */
    public FeatureNode[] buildDocumentVector(Document document, TfIdfFeatureWeighterUtil weighter) {
        int termCount = document.tfMap.size();  // 词的个数
        FeatureNode[] x = new FeatureNode[termCount];//构造特征节点
        Iterator<Map.Entry<Integer, int[]>> tfMapIterator = document.tfMap.entrySet().iterator();//对得到的分词进行遍历,得到特征词和对应的词频
        for (int j = 0; j < termCount; j++) {
            Map.Entry<Integer, int[]> tfEntry = tfMapIterator.next();
            int feature = tfEntry.getKey();//特征词用对应的唯一的特征id标识
            int frequency = tfEntry.getValue()[0];//特征词出现的次数
            x[j] = new FeatureNode(feature + 1, weighter.weight(feature, frequency));//计算TF-IDF
        }
        // 对词向量进行归一化(L2标准化).得到分词后每个词结点的权重值
        double normalizer = 0;
        for (int j = 0; j < termCount; j++) {
            double weight = x[j].getValue();
            normalizer += weight * weight;
        }
        normalizer = Math.sqrt(normalizer);
        for (int j = 0; j < termCount; j++) {
            double weight = x[j].getValue();
            x[j].setValue(weight / normalizer);
        }
        return x;
    }

    /**
     * 根据得到处理后的SVM问题中各个特征参数,求解SVM问题,返回求解完成的模型
     *
     * @param problem svm问题对象
     * @return 生成的model
     */
    public Model solveLibLinearProblem(Problem problem) {
        //选择L1R_LR分类器,L1-regularized logistic regression(L1正则逻辑回归)
        //C 是约束violation的代价参数 （默认为1), eps 是迭代停止条件的容忍度tolerance,均为liblinear包中求解问题的参数
        Parameter lparam = new Parameter(SolverType.L1R_LR, 500., 0.01);
        return Linear.train(problem, lparam);//返回经liblinear训练好的model
    }

    /**
     * 根据text预测分类
     *
     * @param text 输入字符串
     * @return 分类结果
     */
    public String classify(String text) {
        Map<String, Double> scoreMap = this.predict(text);//scoreMap存储分类和对应的权重
        return CollectionUtility.max(scoreMap);//返回value值最大(可能性最大)的分类结果
    }

    /**
     * 预测分类的方法
     *
     * @param text 输入字符串
     * @return 返回分类及其对应权重map
     */
    public Map<String, Double> predict(String text) {
        /*
          分词，新建document存储输入文字
          初始化时调用代参构造方法将输入文字进行分词操作，将字符串数组String[]分割成字符数组char[]
         */
        ITrie<Integer> wordIdTrie = model.wordIdTrie;
        String[] tokenArray = model.tokenizer.segment(text);
        /*Document document = new Document(wordIdTrie,tokenArray);*/
        Document document = new Document(wordIdTrie, tokenArray);
        //WORD_MAP保存特征的id和特征词的对应关系
        for (String word : tokenArray) {
            Integer id = wordIdTrie.get(word.toCharArray());
            WORD_MAP.put(id, word);
        }
        AbstractModel model = this.getModel();
        double[] probs = this.categorize(document);
        Map<String, Double> scoreMap = new HashMap<>();
        for (int i = 0; i < probs.length; ++i) {
            scoreMap.put(model.catalog[i], probs[i]);
        }
        return scoreMap;
    }

    /**
     * 对文档对象进行svm预测分类,返回分类结果权重数组
     *
     * @param document 文档对象
     * @return 分类结果权重数组
     */
    public double[] categorize(Document document) {
        FeatureNode[] x = buildDocumentVector(document, model.featureWeighter);
        double[] probs = new double[7];//保存7个分类试验结果
        //进行SVM预测，得到向量归一化后的数组
        Linear.predictProbability(model.svmModel, x, probs);
        //probs为归一化后的数组
        return probs;
    }
}
