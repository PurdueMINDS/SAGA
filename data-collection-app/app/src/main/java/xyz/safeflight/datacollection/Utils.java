package xyz.safeflight.datacollection;

import android.content.Context;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.concurrent.TimeUnit;

/**
 * Created by migue on 9/22/2016.
 */
public final class Utils {

    public static String formatTime(Long millis) {
        Calendar c= Calendar.getInstance();
        c.setTimeInMillis(millis);

        int hours=c.get(Calendar.HOUR);
        int minutes=c.get(Calendar.MINUTE);
        int seconds = c.get(Calendar.SECOND);
        return (String.valueOf(hours)+":"+String.valueOf(minutes)+":"+String.valueOf(seconds));
    }

    public static String getDate () {
        DateFormat df = new SimpleDateFormat("MM_dd_yy");
        Date dateobj = new Date();
        return df.format(dateobj);
    }

    public static String getTime(String format) { //"HH_mm_ss
        DateFormat df = new SimpleDateFormat(format);
        Date dateobj = new Date();
        return df.format(dateobj);
    }


    public static String getPrettyDirname(String dateString){
        SimpleDateFormat dateFormat = new SimpleDateFormat("MM_dd_yy");
        Date convertedDate = new Date();
        try {
            convertedDate = dateFormat.parse(dateString);
        } catch (ParseException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        dateFormat = new SimpleDateFormat("EEEE, d MMM yyy");

        return dateFormat.format(convertedDate);
    }

    public static String getPrettyFilename(String dateString){
        return dateString.replace("_",":");
    }


    public static File getDocumentsDirectory(Context context, String directory) {
        // Get the directory for the user's public data directory.
        File file;
        if (directory==null){
            file = context.getExternalFilesDir(
                    Environment.DIRECTORY_DOCUMENTS);
        } else {
            file = new File(context.getExternalFilesDir(
                    Environment.DIRECTORY_DOCUMENTS), directory);
        }

        if (!file.mkdirs()) {
            file.mkdir();
        }
        return file;
    }

}
