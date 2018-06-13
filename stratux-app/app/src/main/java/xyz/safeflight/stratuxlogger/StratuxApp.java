package xyz.safeflight.stratuxlogger;

import android.app.Application;

import timber.log.Timber;

public class StratuxApp extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        if (BuildConfig.DEBUG) {
            Timber.plant(new Timber.DebugTree());
        }
    }
}
