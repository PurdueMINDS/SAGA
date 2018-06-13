package xyz.safeflight.stratuxlogger;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.os.Binder;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.support.annotation.Nullable;
import android.support.annotation.RequiresApi;
import android.support.v4.app.NotificationCompat;

import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.LinkedList;
import java.util.List;

import timber.log.Timber;


public class StratuxCollectorService extends Service {
    // Used by the notification
    private static final String CHANNEL_ID = "stratux_collector_service";
    private static final int NOTIFICATION_ID = 1;

    /**
     * Class used for the client Binder.  Because we know this service always
     * runs in the same process as its clients, we don't need to deal with IPC.
     */
    class StratuxCollectorBinder extends Binder {
        StratuxCollectorService getService() {
            // Return this instance of LocalService so clients can call public methods
            return StratuxCollectorService.this;
        }
    }

    /* The client may be bound to the service, but until the service is
     started, no recording is done */
    public enum State {
        COLLECTING,
        NOT_COLLECTING
    }
    private State currentState = State.NOT_COLLECTING;

    // Binder given to clients
    private final IBinder mBinder = new StratuxCollectorBinder();
    Runnable stateChangeCallback;

    private final Object data_lock = new Object();
    List<AHRSData> data = new LinkedList<>();
    File file;
    FileWriter fileWriter;

    int numCollected;
    String lastReceived = "--";

    public void registerCallback(Runnable callback) {
        stateChangeCallback = callback;
    }

    public boolean isCollecting() {
        return currentState == State.COLLECTING;
    }

    public int getNumCollected() { return numCollected; }

    public String getLastReceived() { return lastReceived; }

    private void runCallback() {
        if (stateChangeCallback != null) {
            stateChangeCallback.run();
        }
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return mBinder;
    }

    public void startCollecting() {
        if (currentState == State.NOT_COLLECTING) {
            Intent serviceIntent = new Intent(this, StratuxCollectorService.class);
            startService(serviceIntent);
        }
    }

    public void stopCollecting() {
        Timber.d("stopCollecting()");
        if (currentState == State.COLLECTING) {
            stopForeground(true);
            stopSelf();
            currentState = State.NOT_COLLECTING;

            dumpToFile();

            try {
                fileWriter.close();
            } catch (IOException e) {
                Timber.e(e, "Failed to close CSV file");
            }
            runCallback();
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Timber.d("onStartCommand()");

        // Create a channel for the notification
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            createChannel();
        }

        // Configure the intent to bring the app back when the notification is pressed
        Intent showAppIntent = new Intent(this, MainActivity.class);
        showAppIntent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
        showAppIntent.addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP);
        PendingIntent contentIntent = PendingIntent.getActivity(
                this,
                (int) System.currentTimeMillis(),
                showAppIntent,
                PendingIntent.FLAG_UPDATE_CURRENT
        );

        // Build the notification
        NotificationCompat.Builder builder = new NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.app_name))
            .setContentText("Collecting Stratux data in the background")
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setSmallIcon(R.drawable.ic_stat_collecting)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setOngoing(true)
            .setContentIntent(contentIntent)
            .setVibrate(new long[]{300, 400, 500, 400, 300, 400, 500, 400, 500})
            .setAutoCancel(true);
        Notification notification = builder.build();

        // Set this service as foreground service
        startForeground(NOTIFICATION_ID, notification);

        // Create the CSV file
        boolean fileWriteFailed = false;
        try {
            file = Utils.getAHRSNowFile();
            fileWriter = new FileWriter(file, false);
            AHRSData.dumpCSV(fileWriter, null, true);
        } catch (IOException e) {
            fileWriteFailed = true;
            Timber.e(e, "Failed while creating CSV file");
        }

        if (!fileWriteFailed) {
            // Change our state to: Collecting
            currentState = State.COLLECTING;
            numCollected = 0;
            runCallback();

            // Post the background work
            Handler handler = new Handler(getMainLooper());
            int delay = 900;
            StratuxFetchClass fetchTask = new StratuxFetchClass(this, delay);
            handler.postDelayed(fetchTask, delay);
        }

        return START_NOT_STICKY;
    }

    private void addAHRS(JSONObject ahrs) {
        if (ahrs != null) {
            try {
                AHRSData payload = new AHRSData(ahrs);
                synchronized (data_lock) {
                    data.add(payload);
                    Timber.d("Added to list!");
                }
                numCollected++;
                lastReceived = payload.GPSTime;
                Timber.d("[%d] GPS: (%.4f, %.4f) %s", data.size(), payload.GPSLatitude, payload.GPSLongitude, payload.GPSTime);
                runCallback();

                if (data.size() > 30) {
                    dumpToFile();
                }
            } catch (JSONException e) {
                Timber.e(e, "Error while writing data to file");
            }
        }
    }

    private void dumpToFile() {
        if (fileWriter == null) {
            Timber.w("Tried to write data, but stream was null.");
            return;
        }

        synchronized (data_lock) {
            Timber.d("Dumping %d records.", data.size());
            AHRSData.dumpCSV(fileWriter, data);
            data.clear();
            Timber.d("New size: %d", data.size());
        }
    }

    @RequiresApi(Build.VERSION_CODES.O)
    private void createChannel() {
        NotificationManager mNotificationManager =
                (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);

        // The id of the channel.

        // The user-visible name of the channel.
        CharSequence name = getString(R.string.channel_name);

        // The user-visible description of the channel.
        String description = getString(R.string.channel_description);

        int importance = NotificationManager.IMPORTANCE_DEFAULT;
        NotificationChannel mChannel = new NotificationChannel(CHANNEL_ID, name, importance);

        // Configure the notification channel.
        mChannel.setDescription(description);
        mChannel.enableVibration(true);
        mChannel.setVibrationPattern(new long[]{500, 500, 500, 500});
        mChannel.setLockscreenVisibility(Notification.VISIBILITY_PUBLIC);
        mNotificationManager.createNotificationChannel(mChannel);

        Timber.d("Channel was created");
    }

    private class StratuxFetchClass implements Runnable {

        Handler mHandler;
        int mDelay;
        private WeakReference<StratuxCollectorService> stratuxCollectorServiceWeakReference;
        RequestQueue queue;
        JsonObjectRequest jsObjRequest;

        StratuxFetchClass(StratuxCollectorService service, int delay) {
            stratuxCollectorServiceWeakReference = new WeakReference<>(service);
            mDelay = delay;
            mHandler = new Handler(service.getMainLooper());
            queue = Volley.newRequestQueue(service);
            jsObjRequest = new JsonObjectRequest(
                    Request.Method.GET,
//                    "http://10.0.2.2/getSituation", // Android Emulator (localhost)
                    "http://192.168.10.1/getSituation",
                    null,
                    new Response.Listener<JSONObject>() {
                        @Override
                        public void onResponse(JSONObject response) {

                            StratuxCollectorService service = StratuxFetchClass.this.stratuxCollectorServiceWeakReference.get();
                            if (service != null) {
                                service.addAHRS(response);
                            }
                        }
                    },
                    new Response.ErrorListener() {
                        @Override
                        public void onErrorResponse(VolleyError error) {
                            Timber.e(error, "Failed to retrieve Stratux data:");
                        }
                    }
            );
            jsObjRequest.setRetryPolicy(new DefaultRetryPolicy(
                    (int) (delay * 0.9),
                    0,
                    1)
            );

        }

        @Override
        public void run() {

            // If (for some weird reason) the service does not exist anymore, we don't do anything
            final StratuxCollectorService service = stratuxCollectorServiceWeakReference.get();
            if (service == null) return;

            // If we are still collecting, reschedule task for the future
            if (service.isCollecting()) {
                mHandler.postDelayed(this, mDelay);

                // Add the request to the RequestQueue.
                queue.add(jsObjRequest);
            }

        }
    }

}
