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
        try {
            // true表示不覆盖原来的内容，而是加到文件的后面。若要覆盖原来的内容，直接省略这个参数就好
            fw = new FileWriter(path, false);
            fw.write(data);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                fw.flush();
                fw.close();
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
        fr.close();
        return sb.toString();
    }

    public static void traverseFile(String path) {
        File file = new File(path);

        File[] files = file.listFiles();
        for (File f : files) {
            if (f.isDirectory()) {
                traverseFile(f.getAbsolutePath());
            } else {
                String key = f.getAbsolutePath();
                String[] split = key.split("\\\\");
                String value = split[6];
                map.put(key, value);
            }
        }
        /*Map<String, String> map = new LinkedHashMap<>();
        LinkedList<File> list = new LinkedList<>();

        if (file.exists()) {
            if (null == file.listFiles()) {
                return null;
            }
            list.addAll(Arrays.asList(file.listFiles()));
            while (!list.isEmpty()) {
                File[] files = list.removeFirst().listFiles();
                if (null == files) {
                    continue;
                }
                for (File f : files) {
                    String s = f.getAbsolutePath();
                    String[] split = s.split("\\\\");
                    String category = split[6];
                    map.put(s, category);
                }
            }
        } else {
            return null;
        }
        return map;*/
    }

    public static void main(String[] args) {
        traverseFile("data/搜狗文本分类语料库微型版/测试集");
        Set<Map.Entry<String, String>> set = map.entrySet();
        Iterator<Map.Entry<String, String>> it = set.iterator();
        while (it.hasNext()) {
            Map.Entry<String, String> entry = it.next();
            System.out.println(entry.getKey() + "=" + entry.getValue());
        }
    }
}
