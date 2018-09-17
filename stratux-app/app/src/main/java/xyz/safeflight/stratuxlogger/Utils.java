/*
 *  Copyright 2018 Jianfei Gao, Leonardo Teixeira, Bruno Ribeiro.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

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
