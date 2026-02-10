package org.ironman.framework.bean.app;

import android.os.Build;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class IntentFilter {

    public List<String> actions;
    public List<String> categories;
    public List<String> dataSchemes;
    public List<PatternMatcher> dataSchemeSpecificParts;
    public List<IntentFilter.AuthorityEntry> dataAuthorities;
    public List<PatternMatcher> dataPaths;
    public List<String> dataTypes;

    public IntentFilter(android.content.IntentFilter intent) {
        if (intent.countActions() > 0) {
            actions = new ArrayList<>(intent.countActions());
            Iterator<String> it = intent.actionsIterator();
            while (it.hasNext()) {
                actions.add(it.next());
            }
        }

        if (intent.countCategories() > 0) {
            categories = new ArrayList<>(intent.countCategories());
            Iterator<String> it = intent.categoriesIterator();
            while (it.hasNext()) {
                categories.add(it.next());
            }
        }

        if (intent.countDataSchemes() > 0) {
            dataSchemes = new ArrayList<>(intent.countDataSchemes());
            Iterator<String> it = intent.schemesIterator();
            while (it.hasNext()) {
                dataSchemes.add(it.next());
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            if (intent.countDataSchemeSpecificParts() > 0) {
                dataSchemeSpecificParts = new ArrayList<>(intent.countDataSchemeSpecificParts());
                Iterator<android.os.PatternMatcher> it = intent.schemeSpecificPartsIterator();
                while (it.hasNext()) {
                    dataSchemeSpecificParts.add(new PatternMatcher(it.next()));
                }
            }
        }

        if (intent.countDataAuthorities() > 0) {
            dataAuthorities = new ArrayList<>(intent.countDataAuthorities());
            Iterator<android.content.IntentFilter.AuthorityEntry> it = intent.authoritiesIterator();
            while (it.hasNext()) {
                dataAuthorities.add(new AuthorityEntry(it.next()));
            }
        }

        if (intent.countDataPaths() > 0) {
            dataPaths = new ArrayList<>(intent.countDataPaths());
            Iterator<android.os.PatternMatcher> it = intent.pathsIterator();
            while (it.hasNext()) {
                dataPaths.add(new PatternMatcher(it.next()));
            }
        }

        if (intent.countDataTypes() > 0) {
            dataTypes = new ArrayList<>(intent.countDataTypes());
            Iterator<String> it = intent.typesIterator();
            while (it.hasNext()) {
                dataTypes.add(it.next());
            }
        }

    }

    public static class AuthorityEntry {

        public String host;
        public int port;

        public AuthorityEntry(android.content.IntentFilter.AuthorityEntry authority) {
            host = authority.getHost();
            port = authority.getPort();
        }
    }
}
