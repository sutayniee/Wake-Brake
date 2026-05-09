class ParsingUtils {
  // Enforce whitespace insignificance in parsing as required
  static String cleanWhitespace(String jsonString) {
    return jsonString.replaceAll(RegExp(r'\s+(?=(?:[^"]*"[^"]*")*[^"]*$)'), '');
  }
  
  static const List<String> supportedOperators = ['==', '!=', '>', '<', '>=', '<='];
}
