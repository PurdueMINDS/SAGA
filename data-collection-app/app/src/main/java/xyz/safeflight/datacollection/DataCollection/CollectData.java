package xyz.safeflight.datacollection.DataCollection;

import android.Manifest;
import android.app.Notification;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.os.Parcelable;
import android.support.v4.app.ActivityCompat;
import android.support.v4.app.TaskStackBuilder;
import android.support.v4.content.ContextCompat;
import android.support.v4.content.LocalBroadcastManager;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.app.NotificationCompat;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import xyz.safeflight.datacollection.Constants;
import xyz.safeflight.datacollection.FileManagement.WriteThread;
import xyz.safeflight.datacollection.PlotData.PlotActivity;
import xyz.safeflight.datacollection.R;
import xyz.safeflight.datacollection.DataCollection.Sensors.SensorClass;

import java.util.ArrayList;

public class CollectData extends AppCompatActivity {
    private static final String TAG = "CollectData";
    private BroadcastReceiver receiver;
    private Intent serviceIntent;
    private LocalBroadcastManager sender;
    private boolean resumeHasRun;
    private int lastIndex;
    private TextView counter_display;
    private int counts= 0;
    private ArrayList data_list;
    private final int PERMISSIONS_REQUEST = 1;
    private ArrayList<ImageView> image_views;
    private boolean listreceived = false;
    private boolean gpsstate = true;
    private boolean hasFinished = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_newdesign);
        resumeHasRun = false;
        lastIndex = 0;

        // Get layout elements
        getImageViews();

        // Show sensor states
        ArrayList sensor_states = SensorClass.sensorInfo(getApplicationContext());
        showSensorState(sensor_states);
        data_list = new ArrayList();

        // Register broadcast sender and receiver
        sender = LocalBroadcastManager.getInstance(this);
        // Register receiver to check location messages
        receiver = new BroadcastReceiver(){
            @Override
            public void onReceive(Context context, Intent intent) {
                processMessage(intent);
            }
        };

        // Add broadcast message filters
        addBdFilters();
        // Create background service for retrieving GPS Location
        Bundle b = getIntent().getExtras();
        int period = b.getInt("period");
        serviceIntent = new Intent(getApplicationContext(),LocationService.class);
        serviceIntent.putExtra("period", period);
        Log.e(TAG,"GPS Service started");
        getApplicationContext().startService(serviceIntent);

    }

    @Override
    protected void onStart() {
        super.onStart();
    }

    @Override
    protected void onStop() {
        if (hasFinished){
            Log.e(TAG, "onStop called after finishing activity... doing nothing onstop method");
            hasFinished = false;
            try{
                if(receiver!=null)
                    LocalBroadcastManager.getInstance(this).unregisterReceiver((receiver));
            }catch(Exception e){
                Log.e(TAG,e.toString());
            }

        } else {
            // If gps is enabled
            if (gpsstate) {
                // Tell service onStop has started
                Log.e(TAG, "onStop Activity message sent");
                sendOnStopState();
            } else {
                Log.e(TAG, "onStop no gps found killing service...");
                this.stopService(serviceIntent);
                lastIndex = 0;
            }
        }
        super.onStop();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!resumeHasRun) {
            resumeHasRun = true;
            return;
        }
        Log.e(TAG,"onResume was called send backtowork");
        // Tell the Service that activity is back
        Intent intent = new Intent(Constants.BACK_TO_WORK);
        sender.sendBroadcast(intent);

    }

    @Override
    public void onBackPressed() {
        this.stopService(serviceIntent);
        data_list.clear();
        lastIndex = 0;
        hasFinished = true;
        finish();
        return;
    }

    private void addBdFilters(){
        Log.e(TAG,"WTF AGAIN ADDING FILTERS AFTER CALLING ON CREATE");
        IntentFilter myfilter = new IntentFilter();
        myfilter.addAction(Constants.TURN_ON_LOC);
        myfilter.addAction(Constants.RESULT);
        myfilter.addAction(Constants.RESUME);

        if (checkLocationPermission()==false){
            askLocationPermissions();
        }

        // Register the receiver
		LocalBroadcastManager.getInstance(this).registerReceiver((receiver),myfilter);
    }

    public void getImageViews(){
        image_views = new ArrayList<>();
        ImageView gps_im = (ImageView) findViewById(R.id.gps_image);
        ImageView acc_im = (ImageView) findViewById(R.id.acc_image);
        ImageView bar_im = (ImageView) findViewById(R.id.bar_image);
        ImageView gyr_im = (ImageView) findViewById(R.id.gyr_image);
        counter_display = (TextView) findViewById(R.id.counts);
        image_views.add(gps_im);
        image_views.add(acc_im);
        image_views.add(bar_im);
        image_views.add(gyr_im);
    }

    public void handleStopClick(View v) {
        v.getContext().stopService(serviceIntent);
        data_list.clear();
        lastIndex = 0;
        hasFinished = true;
        finish();
        return;
    }


    private void processMessage(Intent intent){
        String action = intent.getAction();
        switch (action){
            case Constants.TURN_ON_LOC:
                gpsstate = false;
                Log.e(TAG,"Request turn on GPS");
                // Show gps is off in layout
                changeGpsState(1);
                buildAlertMessageNoGps();
                break;
            case Constants.RESULT:
                gpsstate = true;
                Log.e(TAG,"New location received . . . ");
                DataObject data = intent.getParcelableExtra(Constants.MESSAGE);
                // Update GUI with new location
                processData(data);
                break;
            case Constants.RESUME:
                    if (!listreceived) {
                        Log.e(TAG, "List of locations received. . .");
                        // Add received locations to local list

						ArrayList<Parcelable> loc_list = intent.getParcelableArrayListExtra(Constants.MESSAGE);
						if (loc_list != null) {
							Log.i(TAG, loc_list.toString());
							data_list.addAll(loc_list);
							writeList();
						}
                        listreceived=true;
                    }
                break;
        }
    }


    private void processData(DataObject data){
        data_list.add(data);
        // Display in screen
        if (data_list.size()>=Constants.MAX_LOCATIONS) {
            writeList();
        } else {
            counts += 1;
        }
        counter_display.setText(String.valueOf(counts));

    }

    private void writeList(){
        Log.e(TAG,"Starting thread for writing...");
        ArrayList<DataObject> copy = (ArrayList<DataObject>) this.data_list.clone();

        // Write to file
        Runnable r = new WriteThread(this,copy,lastIndex);
        new Thread(r).start();

        // Reset list
        lastIndex += data_list.size();
        counts = lastIndex;
        this.data_list.clear();
        counter_display.setText(String.valueOf(counts));
    }

    private void showSensorState(ArrayList<Integer> states) {
        for (int i=0; i< 4; i++){
            if (states.get(i)==1){
                image_views.get(i).setImageResource(R.drawable.miss);
            }
        }
    }

    private void changeGpsState(Integer state){
        if (state==0){
            image_views.get(0).setImageResource(R.drawable.check);
        } else {
            image_views.get(0).setImageResource(R.drawable.miss);
        }
    }

    private void buildAlertMessageNoGps() {
        final AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setMessage("Your GPS seems to be disabled, do you want to enable it?")
                .setCancelable(false)
                .setPositiveButton("Yes", new DialogInterface.OnClickListener() {
                    public void onClick(@SuppressWarnings("unused") final DialogInterface dialog, @SuppressWarnings("unused") final int id) {
                        dialog.cancel();
                        finish();
                        startActivity(new Intent(android.provider.Settings.ACTION_LOCATION_SOURCE_SETTINGS));
                    }
                })
                .setNegativeButton("No", new DialogInterface.OnClickListener() {
                    public void onClick(final DialogInterface dialog, @SuppressWarnings("unused") final int id) {
                        dialog.cancel();
                        Toast.makeText(getApplicationContext(), "You need to activate your GPS to use this app", Toast.LENGTH_LONG).show();
                        finish();
                    }
                });
        final AlertDialog alert = builder.create();
        if (! this.isFinishing()) {
            alert.show();
        }
    }

    public void sendOnStopState(){
        Log.e(TAG,"Broadcast message send to service...");
        Intent intent = new Intent(Constants.NO_COMMUNICATION);
        sender.sendBroadcast(intent);
        listreceived = false;
        Log.e(TAG,"Done");
        createNotification();
    }

    public void sendPlotState() {
        Log.e(TAG, "Broadcasting plotting message!");
        Intent intent = new Intent(Constants.PLOT);
        sender.sendBroadcast(intent);
        listreceived = false;

    }

    public void askLocationPermissions(){
        // Here, thisActivity is the current activity
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.ACCESS_FINE_LOCATION)
                != PackageManager.PERMISSION_GRANTED) {
                Log.e(TAG,"ASKING PERMISSIONS");
                // No explanation needed, we can request the permission.
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.ACCESS_FINE_LOCATION},
                        PERMISSIONS_REQUEST);

        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String permissions[], int[] grantResults) {
        switch (requestCode) {
            case PERMISSIONS_REQUEST: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                } else {
                    finish();
                    // permission denied, boo! Disable the
                    // functionality that depends on this permission.
                }
                return;
            }
        }
    }

    public boolean checkLocationPermission()
    {
        String permission = "android.permission.ACCESS_FINE_LOCATION";
        int res = this.checkCallingOrSelfPermission(permission);
        return (res == PackageManager.PERMISSION_GRANTED);
    }

    // Handle screen rotation
    @Override
    public void onConfigurationChanged(Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
        // Checks the orientation of the screen
        if (newConfig.orientation == Configuration.ORIENTATION_LANDSCAPE) {
            setContentView(R.layout.activity_nd_landscape);
        } else if (newConfig.orientation == Configuration.ORIENTATION_PORTRAIT){
            setContentView(R.layout.activity_newdesign);
        }
        getImageViews();
        counter_display.setText(String.valueOf(counts));
    }

    private void createNotification(){
        NotificationCompat.Builder mBuilder =
                (NotificationCompat.Builder) new NotificationCompat.Builder(this)
                        .setSmallIcon(R.drawable.plane_icon)
                        .setContentTitle("Collecting data")
                        .setContentText("Safe flight is now collecting data.");

        // Creates an explicit intent for an Activity in your app
        Intent notificationIntent = new Intent(this, CollectData.class);
        // Necessary to launch the last activity after pressing the notification
        notificationIntent.setAction(Intent.ACTION_MAIN);
        notificationIntent.addCategory(Intent.CATEGORY_LAUNCHER);
        notificationIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);

        // Pending Intent to give permission to launch
        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0,
                notificationIntent, 0);
        mBuilder.setContentIntent(pendingIntent);

        NotificationManager mNotificationManager =
                (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);

        // Only necessary to make the notification dissapear on foreground
        mBuilder.setAutoCancel(true);
        mBuilder.getNotification().flags |= Notification.FLAG_AUTO_CANCEL;

        // mId allows you to update the notification later on.
        mNotificationManager.notify(333, mBuilder.build());
    }

    public void onClickPlot(View v) {
        if (hasFinished){
            Toast.makeText(getApplicationContext(), "Please start data collection first", Toast.LENGTH_LONG).show();

        } else {
            // If gps is enabled
            if (gpsstate) {
                // Tell service that must send plotting data
                Log.e(TAG, "onPlot asking service for data...");
                sendPlotState();
                Intent intent = new Intent(this, PlotActivity.class);
                startActivity(intent);
            } else {
                Toast.makeText(getApplicationContext(), "Please turn on GPS", Toast.LENGTH_LONG).show();
            }
        }
    }
}
