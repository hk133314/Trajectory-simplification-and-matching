var express = require("express");
var router = express.Router();

/* GET users listing. */
router.get("/", function (req, res, next) {
    res.send("respond with a resource");
});

module.exports = router;

var pool = require("./pg");
router.get("/index2", function (req, res) {
    pool.connect(function (err, client) {
        if (err) {
            console.log(err);
        } else {
            // "SELECT * FROM \"666\".\"road_network2\" order by \"id\" "
            client.query("SELECT * FROM \"666\".\"road_network2\"", function (err, result) {
                var traces_pts = "";
                // var traces_id = "";
                var i;
                var j;
                // for (i = 0; i < 1; i++) {
                for (i = 0; i < result.rows.length; i++) {
                    var trace_text = result.rows[i].all_trace_pts.split("],");
                    // var trace_id = result.rows[i].id;
                    // console.log(trace_text)
                    var trace_pts = [];
                    for (j = 0; j < trace_text.length; j++) {
                        var pos = [];
                        let record;
                        var RegExp_pos = new RegExp(/\d+.\d+/, "g");
                        while (record = RegExp_pos.exec(trace_text[j])) {
                            pos.push(parseFloat(record[0]));
                        }
                        trace_pts.push(pos);
                    }
                    traces_pts += trace_pts.toString() + ",,,";
                    // traces_id += trace_id.toString() + ",,,";
                }
                traces_pts = traces_pts.substr(0, traces_pts.length - 3);
                // traces_id = traces_id.substr(0, traces_id.length - 3);
                res.render("index2", {
                    traces_pts: traces_pts
                });
            });
            client.release();
        }
    });
});

router.post("/index2", function (req, res) {
    var feature_index = req.body.feature_index;
    var feature_pts = req.body.feature_pts;
    var pts_distance = req.body.pts_distance;
    var line_wkts = req.body.line_wkts;
    feature_index = feature_index.split(",,,");
    feature_pts = feature_pts.split(",,,");
    pts_distance = pts_distance.split(",,,");
    line_wkts = line_wkts.split(",,,");
    console.log(feature_index.length);
    console.log(feature_pts.length);
    console.log(pts_distance.length);
    console.log(line_wkts.length);
    pool.connect(function (err, client) {
        if (err) {
            console.log(err);
        } else {
            var delete_sql1 = "TRUNCATE TABLE \"666\".\"simply_traces\"";
            client.query(delete_sql1);
            for (let i = 0; i < feature_index.length; i++) {
                var insert_sql1 = "INSERT INTO \"666\".\"simply_traces\" VALUES(" + (i + 1).toString() + ",'"
                    + feature_index[i] + "','"
                    + feature_pts[i] + "','"
                    + pts_distance[i] + "','"
                    + line_wkts[i] + "')";
                // console.log(insert_sql1)
                client.query(insert_sql1);
            }
            client.release();
        }
    });
    console.log("insert_finished");
});