package edu.upc.svmtraining.util;

import java.io.*;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

public class FileOperationUtil {
    //用来存遍历文件夹之后得到的文件路径和文件分类名
    public static Map<String, String> map = new LinkedHashMap<>();

    public static void writeFile(String path, String data) {
        FileWriter fw = null;
        BufferedWriter bw = null;
        try {
            // true表示不覆盖原来的内容，而是加到文件的后面。若要覆盖原来的内容，直接省略这个参数就好
            fw = new FileWriter(path, false);
            bw = new BufferedWriter(fw);
            bw.write(data);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                assert bw != null;
                bw.flush();
                bw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static String readFile(String path) throws IOException {
        //读取清洗后的数据源
        FileReader fr = new FileReader(path);
        BufferedReader br = new BufferedReader(fr);
        StringBuilder sb = new StringBuilder();

        String line;
        while ((line = br.readLine()) != null) {
            sb.append(line);
        }
        br.close();
        return sb.toString();
    }

    public static void traverseFile(String path) {
        File file = new File(path);
        File[] files = file.listFiles();
        assert files != null;
        for (File f : files) {
            if (f.isDirectory()) {
                traverseFile(f.getAbsolutePath());
            } else {
                String key = f.getAbsolutePath();
                String[] split = key.split("\\\\");
                String value = split[7];//需要根据自己路径自行定义
                map.put(key, value);
            }
        }
    }

    public static void main(String[] args) {
        traverseFile("data/搜狗文本分类语料库微型版/测试集");
        Set<Map.Entry<String, String>> set = map.entrySet();
        for (Map.Entry<String, String> entry : set) {
            System.out.println(entry.getKey() + "——>" + entry.getValue());
        }
    }
}
