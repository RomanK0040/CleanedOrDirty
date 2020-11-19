package com.example.cleanedordirty;

import android.content.Context;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class Utils {
    public static String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static double[] softmax(float[] scores) {
        double[] e_score = new double[scores.length];
        double e_scoreSum = 0;

        double[] softmaxOutput = new double[scores.length];
        for (int i = 0; i<scores.length; i++) {
           e_score[i] = Math.exp(scores[i]);
           e_scoreSum += e_score[i];
        }

        for (int i = 0; i<e_score.length; i++) {
            softmaxOutput[i] = e_score[i] / e_scoreSum;
        }

        return softmaxOutput;
    }
}
