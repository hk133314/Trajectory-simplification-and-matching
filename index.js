var express = require("express");
var router = express.Router();

router.get("/", function (req, res, next) {
    res.render("index2", {traces_pts: ""});
});

// 要想使用ajax发送post请求，必须加这个
router.post("/", function (req, res, next) {
    res.render("index2");
});

module.exports = router;
