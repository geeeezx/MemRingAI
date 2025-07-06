# Intent Recognition Data

This directory contains the data definitions and configurations for the intent recognition system.

## Files

### `categories.json`
Contains the intent category definitions with:
- **Description**: What each intent category represents
- **Examples**: Sample texts that belong to each category
- **Keywords**: Key terms associated with each intent
- **Patterns**: Common phrase patterns for each intent

### `rules.json`
Contains the rule engine configuration with:
- **Priority**: Processing order for intent rules
- **Conditions**: Rules for matching text to intents
- **Weights**: Confidence calculation weights
- **Minimum Confidence**: Threshold for accepting matches

## Intent Categories

Current supported intent categories:
- `question` - Questions seeking information
- `request` - Requests for action or service
- `complaint` - Expressions of dissatisfaction
- `compliment` - Expressions of satisfaction
- `booking` - Reservation requests
- `cancellation` - Cancellation requests
- `information` - Information seeking
- `support` - Help requests
- `other` - Default category for unclear intents

## Rule Engine

The rule engine supports several condition types:
- `contains_any` - Text contains any of the specified values
- `contains_phrase` - Text contains specific phrases
- `starts_with` - Text starts with specified values
- `ends_with` - Text ends with specified values
- `regex_match` - Text matches regex pattern (TODO)
- `length_range` - Text length within range (TODO)
- `word_count` - Word count meets criteria (TODO)

## Usage

These files are automatically loaded by the IntentService when the application starts.

## TODO

- [ ] Add support for regex patterns
- [ ] Add support for entity extraction
- [ ] Add support for contextual rules
- [ ] Add support for learning from user feedback
- [ ] Add support for multiple languages
- [ ] Add support for custom categories
- [ ] Add data validation utilities
- [ ] Add data update/reload mechanisms 