package xyz.safeflight.stratuxlogger;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.Locale;
import java.util.TimeZone;

import timber.log.Timber;

public class Utils {
    /* Checks if external storage is available for read and write */
    public static boolean isExternalStorageWritable() {
        String state = Environment.getExternalStorageState();
        return Environment.MEDIA_MOUNTED.equals(state);
    }

    static File getAHRSStorageDir() {
        // Get the directory
        File dir = new File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS),
                "STRATUX_DATA"
        );

        if (!dir.exists() && !dir.mkdirs()) {
            Timber.e("Failed to create directory");
        }
        return dir;
    }

    static File getAHRSNowFile() {
        Date time = Calendar.getInstance().getTime();
        SimpleDateFormat outputFmt = new SimpleDateFormat("yyyy_MM_dd-HH_mm_ss", Locale.US);
//        outputFmt.setTimeZone(TimeZone.getTimeZone("UTC"));
        String dateAsString = outputFmt.format(time) + ".csv";

        File dir = getAHRSStorageDir();

        return new File(dir, dateAsString);
    }
}
