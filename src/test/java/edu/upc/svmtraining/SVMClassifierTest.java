package edu.upc.svmtraining;

import edu.upc.svmtraining.classifier.SVMClassifier;
import edu.upc.svmtraining.model.SVMModel;
import junit.framework.TestCase;

import java.io.*;
import java.util.Map;

import static com.hankcs.hanlp.utility.Predefine.logger;

public class SVMClassifierTest extends TestCase {
    public static final String CORPUS_FOLDER = "data/搜狗文本分类语料库微型版/训练集";
    /**
     * 模型保存路径
     */
    public static final String MODEL_PATH = "data/svm-classification-model.ser";

    public static void main(String[] args) throws IOException {
        SVMClassifier classifier = new SVMClassifier(trainOrLoadModel());
        String predict = predict(classifier, "电子,货币,电子,发布");
        System.out.println(predict);
        for (Map.Entry<Integer, String> entry : classifier.WORD_MAP.entrySet()) {
            System.out.println(entry.getKey() + "=" + entry.getValue());
        }
    }

    private static String predict(SVMClassifier classifier, String text) {
        /*
          LinearSVMClassifier是最顶层的接口
          调用AbstractClassifier的classify方法（多态）,将文本作为参数预测去对比分类,
          对于不同分类结果和预期特征值用Map<String,Double>来存
          最终返回的是经过特征向量比对后此map中得分最高的分类
         */
        return classifier.classify(text);
    }

    private static SVMModel trainOrLoadModel() throws IOException {
        SVMModel model = (SVMModel) readObjectFrom(MODEL_PATH);
        if (model != null) {
            return model;
        }
        SVMClassifier classifier = new SVMClassifier();  // 创建分类器
        classifier.train(CORPUS_FOLDER, "UTF-8"); // 训练后的模型支持持久化
        model = (SVMModel) classifier.getModel();
        saveObjectTo(model, MODEL_PATH);
        return model;
    }

    /**
     * 序列化对象
     *
     * @param o    存储对象
     * @param path 存储路径
     */
    public static void saveObjectTo(Object o, String path) {
        try {
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path));
            oos.writeObject(o);
            oos.close();
        } catch (IOException e) {
            logger.warning("在保存对象" + o + "到" + path + "时发生异常" + e);
        }

    }

    /**
     * 反序列化对象
     *
     * @param path 读取路径
     * @return 读取模型
     */
    public static Object readObjectFrom(String path) {
        ObjectInputStream ois = null;
        try {
            ois = new ObjectInputStream(new FileInputStream(path));
            Object o = ois.readObject();
            ois.close();
            return o;
        } catch (Exception e) {
            logger.warning("在从" + path + "读取对象时发生异常" + e);
        }

        return null;
    }
}