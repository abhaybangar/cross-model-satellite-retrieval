const mongoose = require("mongoose");

const QueryLogSchema = new mongoose.Schema({
  queryFilename: {
    type: String,
    required: true,
  },
  results: [
    {
      filename: String,
      score: Number,
    },
  ],
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.models.QueryLog || mongoose.model("QueryLog", QueryLogSchema);
