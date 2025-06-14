export default {
  expo: {
    name: 'GenCare',
    slug: 'gencare',
    version: '3.0.0',
    icon: "./assets/Logo/baby-boy.png",
    sdkVersion: "53.0.0",
    extra: {
      apiUrl: 'http://192.168.1.2:5000',
      // apiUrl: 'https://gen-care-backend.vercel.app',
      
      eas: {
        projectId: "806cd55a-e5b4-48df-acf8-e9d0cb13847a"
      }
    },
    android: {
      package: "com.ahmed_shams2.gencare",
      adaptiveIcon: {
        foregroundImage: "./assets/Logo/baby-boy.png",
        backgroundColor: "#ffffff"
      },
      versionCode : 3,
    },
    ios: {
      bundleIdentifier: "com.ahmed-shams2.gencare"
    },
    owner: "ahmed_shams2",
    runtimeVersion: {
      policy: "sdkVersion"
    },
    updates: {
      url: "https://u.expo.dev/806cd55a-e5b4-48df-acf8-e9d0cb13847a"
    },
    assetBundlePatterns: [
      "assets/Logo/*",
      "assets/diseases/*",
      "assets/gif/*"
    ],
    plugins: [
      "expo-router",
      "expo-secure-store"
    ],
    newArchEnabled: true, 
    scheme: "gencare"
  }
};