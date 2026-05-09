import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = 'http://192.168.1.10:5000';

  static Future<void> panicReset() async {
    try {
      await http.post(Uri.parse('$baseUrl/panic-reset'));
    } catch (e) {
      // ignore: avoid_print
      debugPrint("Panic reset failed: $e");
    }
  }
}
