package com.example.cleanedordirty;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.v4.os.IResultReceiver;
import android.util.Size;
import android.widget.TextView;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_CODE_CAMERA_PERMISSION = 200;

    private ExecutorService cameraExecutor;
    private PreviewView previewView;
    private TextView probabilityView;
    private TextView resultView;


    private Module mModule;
    private FloatBuffer mInputTensorBuffer;
    private Tensor mInputTensor;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        resultView = findViewById(R.id.analyseResultField);
        probabilityView = findViewById(R.id.analyseProbabilityField);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            setupCamera();
        } else {
            requestPermissions(new String[] {Manifest.permission.CAMERA}, REQUEST_CODE_CAMERA_PERMISSION);
        }

        cameraExecutor = Executors.newSingleThreadExecutor();
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                Toast.makeText(
                        this,
                        "You can't use image classification app without granting CAMERA permission",
                        Toast.LENGTH_SHORT)
                        .show();
                finish();
            } else {
                setupCamera();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
    }

    private void setupCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();

                preview.setSurfaceProvider(previewView.createSurfaceProvider());


                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(224, 224))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(cameraExecutor, image -> {
                    int rotationDegrees = image.getImageInfo().getRotationDegrees();
                    double result = analyzeImage(image, rotationDegrees);
                    runOnUiThread(() -> updateResult(result));
                    image.close();
                });

                Camera camera = cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalysis, preview);
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));

    }

    private void updateResult(double result) {
        if (result > 0.5) {
            resultView.setText("Dirty");
        } else {
            resultView.setText("Cleaned");
        }
        String percentResult = String.format(Locale.US, "%.2f", (result * 100)) + " %";
        probabilityView.setText(percentResult);
    }


    @SuppressLint("UnsafeExperimentalUsageError")
    private double analyzeImage(ImageProxy image, int rotationDegrees) {
        String moduleAssetName = "model.pt";
        try {
            if (mModule == null) {
                final String moduleFileAbsoluteFilePath = new File(Utils.assetFilePath(this, moduleAssetName)).getAbsolutePath();
                mModule = Module.load(moduleFileAbsoluteFilePath);

                mInputTensorBuffer = Tensor.allocateFloatBuffer(3*224*224);
                mInputTensor = Tensor.fromBlob(mInputTensorBuffer, new long[]{1, 3, 224, 224});
            }

            TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
                    image.getImage(), rotationDegrees, 224, 224,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB,
                    mInputTensorBuffer, 0);

            final Tensor outputTensor = mModule.forward(IValue.from(mInputTensor)).toTensor();
            final float[] scores = outputTensor.getDataAsFloatArray();
            return Utils.softmax(scores)[1];
        } catch (Exception ex) {
            ex.printStackTrace();
            return 0;
        }
    }
}