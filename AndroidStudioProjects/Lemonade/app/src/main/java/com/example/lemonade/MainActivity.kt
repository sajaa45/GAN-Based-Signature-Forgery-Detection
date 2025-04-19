package com.example.lemonade

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.lemonade.R
import com.example.lemonade.ui.theme.LemonadeTheme
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            LemonadeTheme {
                LemonApp()
            }
        }
    }
}
@Composable
fun LemonApp() {
    var currentStep by remember { mutableStateOf(1) }
    var squeezeCount by remember { mutableStateOf(0) }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        when (currentStep) {
            1 -> {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center,
                    modifier = Modifier.fillMaxSize()
                ) {
                    Text(text = stringResource(R.string.one),
                        fontSize = 18.sp)
                    Spacer(modifier = Modifier.height(32.dp))
                    Image(
                        painter = painterResource(R.drawable.lemon_tree),
                        contentDescription = stringResource(R.string.Lemon_tree),
                        modifier = Modifier
                            .wrapContentSize()
                            .border(
                                width = 2.dp,
                                color = Color(105, 205, 216),
                                shape = RoundedCornerShape(4.dp)
                            )
                            .background(
                                color = Color(105, 205, 216),
                                shape = RoundedCornerShape(4.dp)
                            )
                            .padding(8.dp)
                            .clickable {
                                // Move to step 2 and set squeeze count randomly
                                squeezeCount = (2..4).random()
                                currentStep = 2
                            }
                    )

                }
            }

            2 -> {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center,
                    modifier = Modifier.fillMaxSize()
                ) {
                    Text(text = stringResource(R.string.two),
                        fontSize = 18.sp)
                    Spacer(modifier = Modifier.height(32.dp))
                    Image(
                        painter = painterResource(R.drawable.lemon_squeeze),
                        contentDescription = stringResource(R.string.Lemon),
                        modifier = Modifier
                            .wrapContentSize()
                            .border(
                                width = 2.dp,
                                color = Color(105, 205, 216),
                                shape = RoundedCornerShape(4.dp)
                            )
                            .background(
                                color = Color(105, 205, 216),
                                shape = RoundedCornerShape(4.dp)
                            )
                            .padding(8.dp)
                            .clickable {
                                squeezeCount--
                                if (squeezeCount == 0) {
                                    currentStep = 3
                                }
                            }
                    )
                }
            }

            3 -> {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center,
                    modifier = Modifier.fillMaxSize()
                ) {
                    Text(text = stringResource(R.string.three),
                        fontSize = 18.sp)
                    Spacer(modifier = Modifier.height(32.dp))
                    Image(
                        painter = painterResource(R.drawable.lemon_drink),
                        contentDescription = stringResource(R.string.Glass_lemonade),
                        modifier = Modifier
                            .wrapContentSize()
                            .border(
                                width = 2.dp,
                                color = Color(105, 205, 216),
                                shape = RoundedCornerShape(4.dp)
                            )
                            .background(
                                color = Color(105, 205, 216),
                                shape = RoundedCornerShape(4.dp)
                            )
                            .padding(8.dp)
                            .clickable {
                                currentStep = 4
                            }
                    )
                }
            }

            4 -> {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center,
                    modifier = Modifier.fillMaxSize()
                ) {
                    Text(text = stringResource(R.string.four),
                        fontSize = 18.sp)
                    Spacer(modifier = Modifier.height(32.dp))
                    Image(
                        painter = painterResource(R.drawable.lemon_restart),
                        contentDescription = stringResource(R.string.Empty_glass),
                        modifier = Modifier
                            .wrapContentSize()
                            .border(
                                width = 2.dp,
                                color = Color(105, 205, 216),
                                shape = RoundedCornerShape(4.dp)
                            )
                            .background(
                                color = Color(105, 205, 216),
                                shape = RoundedCornerShape(4.dp)
                            )
                            .padding(8.dp)
                            .clickable {
                                currentStep = 1
                            }
                    )
                }
            }
        }
    }
}


@Preview(showBackground = true)
@Composable
fun DefaultPreview() {
    LemonadeTheme {
        LemonApp()
    }
}